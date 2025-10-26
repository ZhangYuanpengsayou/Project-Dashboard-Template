import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

PG_SCHEMA = os.getenv("PG_SCHEMA", "public")


def qualify(sql: str) -> str:
    return sql.replace("{S}.", f"{PG_SCHEMA}.")


CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv(
            "PG_URI",
            "postgresql+psycopg2://postgres:541011@localhost:5432/Smart Campus Transportation",
        ),
        "queries": {
            "User: remaining seats for trip (table)": {
                "sql": """
                    SELECT v.max_capacity - COUNT(t.seat_no) AS remaining_seats
                    FROM   {S}.Vehicle v
                    JOIN   {S}.Trip    tr ON tr.vehicle_id = v.vehicle_id
                    LEFT   JOIN {S}.Ticket t
                           ON t.trip_id = tr.trip_id
                          AND t.status = 'VALID'
                    WHERE  tr.trip_id = :trip_id
                    GROUP  BY v.max_capacity;
                """,
                "chart": {"type": "table"},
                "tags": ["user"],
                "params": ["trip_id"],
            },
            "User: total personal cost in last 30 days (table)": {
                "sql": """
                    SELECT SUM(price) AS total_spent
                    FROM   {S}.Ticket
                    WHERE  user_id = :user_id
                      AND  purchase_time >= NOW() - INTERVAL '30 days'
                      AND  status <> 'REFUNDED';
                """,
                "chart": {"type": "table"},
                "tags": ["user"],
                "params": ["user_id"],
            },
            "User: available trips from Stop A to Stop B (table)": {
                "sql": """
                    SELECT tr.trip_id,
                           r.route_name,
                           tr.sched_start,
                           v.plate,
                           v.max_capacity - COUNT(tk.seat_no) AS available_seats
                    FROM   {S}.Trip tr
                    JOIN   {S}.Route r ON tr.route_id = r.route_id
                    JOIN   {S}.Vehicle v ON tr.vehicle_id = v.vehicle_id
                    JOIN   {S}.Route_stop rs1 ON tr.route_id = rs1.route_id
                    JOIN   {S}.Route_stop rs2 ON tr.route_id = rs2.route_id
                    LEFT   JOIN {S}.Ticket tk
                           ON tr.trip_id = tk.trip_id
                          AND tk.status = 'VALID'
                    WHERE  rs1.stop_id = :segment_id      -- 起始站点
                      AND  rs2.stop_id = :segment_id + 2  -- 示例终点站点
                      AND  rs1.seq_in_route < rs2.seq_in_route
                      AND  tr.status = 'PLANNED'
                      AND  tr.sched_start > NOW()
                    GROUP  BY tr.trip_id,
                              r.route_name,
                              tr.sched_start,
                              v.plate,
                              v.max_capacity
                    HAVING v.max_capacity - COUNT(tk.seat_no) > 0;
                """,
                "chart": {"type": "table"},
                "tags": ["user"],
                "params": ["segment_id"],
            },

            "Dispatcher: list all IN_PROGRESS vehicles today (table)": {
                "sql": """
                    SELECT v.vehicle_id,
                           v.plate,
                           v.lon,
                           v.lat,
                           v.updated_at
                    FROM   {S}.Vehicle v
                    JOIN   {S}.Trip t ON v.vehicle_id = t.vehicle_id
                    WHERE  t.status = 'IN_PROGRESS'
                      AND  DATE(t.sched_start) = CURRENT_DATE;
                """,
                "chart": {"type": "table"},
                "tags": ["dispatcher"],
                "params": [],
            },
            "Dispatcher: segments with congestion > 1.8 (table)": {
                "sql": """
                    SELECT segment_id,
                           from_stop_id,
                           to_stop_id,
                           congestion_fact,
                           distance_m
                    FROM   {S}.Road_segment
                    WHERE  congestion_fact > 1.8
                    ORDER  BY congestion_fact DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["dispatcher"],
                "params": [],
            },
            "Dispatcher: real-time location & load of operating vehicles (table)": {
                "sql": """
                    SELECT v.vehicle_id,
                           v.plate,
                           v.lon,
                           v.lat,
                           v.current_load,
                           v.max_capacity,
                           r.route_name,
                           tr.direction,
                           tr.sched_start
                    FROM   {S}.Vehicle v
                    JOIN   {S}.Trip tr ON v.vehicle_id = tr.vehicle_id
                    JOIN   {S}.Route r ON tr.route_id = r.route_id
                    WHERE  tr.status = 'IN_PROGRESS'
                      AND  v.updated_at > NOW() - INTERVAL '5 minutes'
                    ORDER  BY r.route_name,
                              tr.sched_start;
                """,
                "chart": {"type": "table"},
                "tags": ["dispatcher"],
                "params": [],
            },

            "Admin: users with negative balance (table)": {
                "sql": """
                    SELECT user_id,
                           full_name,
                           card_balance
                    FROM   {S}."User"
                    WHERE  card_balance < 0;
                """,
                "chart": {"type": "table"},
                "tags": ["administrator"],
                "params": [],
            },
            "Admin: balance statistics by user type (bar)": {
                "sql": """
                    SELECT role::text AS role_name,
                           COUNT(*) AS user_count,
                           ROUND(AVG(card_balance), 2) AS avg_balance,
                           SUM(CASE WHEN card_balance < 10 THEN 1 ELSE 0 END) AS low_balance_count
                    FROM   {S}."User"
                    GROUP  BY role
                    ORDER  BY avg_balance DESC;
                """,
                "chart": {"type": "bar", "x": "role_name", "y": "avg_balance"},
                "tags": ["administrator"],
                "params": [],
            },
            "Admin: vehicle utilization & maintenance analysis (table)": {
                "sql": """
                    SELECT v.vehicle_id,
                           v.plate,
                           v.model,
                           v.max_capacity,
                           COUNT(tr.trip_id) AS total_trips,
                           SUM(CASE WHEN tr.status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_trips,
                           ROUND(AVG(v.current_load::numeric / v.max_capacity * 100), 2) AS avg_utilization_percent
                    FROM   {S}.Vehicle v
                    LEFT   JOIN {S}.Trip tr
                           ON v.vehicle_id = tr.vehicle_id
                          AND tr.sched_start >= NOW() - INTERVAL '7 days'
                    GROUP  BY v.vehicle_id,
                              v.plate,
                              v.model,
                              v.max_capacity
                    ORDER  BY avg_utilization_percent DESC NULLS LAST;
                """,
                "chart": {"type": "table"},
                "tags": ["administrator"],
                "params": [],
            },
        },
    },
    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB", "smart_campus_nosql"),
        "queries": {
            "Telemetry: Highest temp per vehicle (last 24h)": {
                "collection": "vehicle_telemetry",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$vehicle_id", "max_temp": {"$max": "$temp"}}},
                    {"$sort": {"max_temp": -1}},
                ],
                "chart": {"type": "bar", "x": "_id", "y": "max_temp"},
            },
            "Telemetry: Lowest battery per vehicle (last 24h)": {
                "collection": "vehicle_telemetry",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$vehicle_id", "min_battery": {"$min": "$battery"}}},
                    {"$sort": {"min_battery": 1}},
                ],
                "chart": {"type": "bar", "x": "_id", "y": "min_battery"},
            },
            "Telemetry: Average speed heat-map (last 24h)": {
                "collection": "vehicle_telemetry",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$vehicle_id", "avg_speed": {"$avg": "$speed"}}},
                    {"$sort": {"avg_speed": -1}},
                ],
                "chart": {"type": "bar", "x": "_id", "y": "avg_speed"},
            },

            "Sensor Tech: temperature sensor drifting (table)": {
                "collection": "sensor_reading",
                "aggregate": [
                    {"$match": {"sensor_id": 2, "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=1)}}},
                    {"$group": {"_id": "$sensor_id", "stdDev": {"$stdDevSamp": "$value"}}},
                    {"$match": {"stdDev": {"$gt": 2}}},
                ],
                "chart": {"type": "table"},
            },
            "Sensor Tech: current readings from all sensors (table)": {
                "collection": "sensor_reading",
                "aggregate": [
                    {"$sort": {"ts": -1}},
                    {"$group": {"_id": "$sensor_id", "lastValue": {"$first": "$value"}, "lastTime": {"$first": "$ts"}}},
                ],
                "chart": {"type": "table"},
            },
            "Sensor Tech: current CO₂ level inside vehicle (table)": {
                "collection": "sensor_reading",
                "find": {"sensor_id": 9},
                "projection": {"value": 1, "ts": 1},
                "sort": [("ts", -1)],
                "limit": 1,
                "chart": {"type": "table"},
            },

            "Dispatcher: five most congested road sections now (table)": {
                "collection": "road_congestion_ts",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(minutes=5)}}},
                    {"$group": {"_id": "$segment_id", "avgCF": {"$avg": "$congestion_fact"}}},
                    {"$sort": {"avgCF": -1}},
                    {"$limit": 5},
                ],
                "chart": {"type": "table"},
            },
            "Dispatcher: current location of operating vehicles (table)": {
                "collection": "Vehicle",
                "aggregate": [
                    {"$match": {"vehicle_id": {"$in": []}}},  
                ],
                "pre_hook": "fill_active_vehicles", 
                "chart": {"type": "table"},
            },
            "Dispatcher: vehicles with door left open (table)": {
                "collection": "door_status",
                "find": {"status": "OPEN"},
                "projection": {"vehicle_id": 1, "door": 1, "ts": 1},
                "chart": {"type": "table"},
            },


            "Data Scientist: battery daily average (line)": {
                "collection": "sensor_reading_ts",
                "aggregate": [
                    {"$match": {"sensor_id": 3, "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=30)}}},
                    {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts"}}, "avgBat": {"$avg": "$value"}}},
                    {"$sort": {"_id": 1}},
                ],
                "chart": {"type": "line", "x": "_id", "y": "avgBat"},
            },
            "Data Scientist: total mileage travelled last 24h (table)": {
                "collection": "sensor_reading_ts",
                "aggregate": [
                    {"$match": {"sensor_id": 7, "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=1)}}},
                    {"$group": {"_id": None, "totalKm": {"$sum": "$value"}}},
                ],
                "chart": {"type": "table"},
            },
            "Data Scientist: anomaly count by type last 24h (bar)": {
                "collection": "vehicle_alerts",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=1)}}},
                    {"$group": {"_id": "$alert_type", "cnt": {"$sum": 1}}},
                ],
                "chart": {"type": "bar", "x": "_id", "y": "cnt"},
            },
        },
    },
}


@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)


@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None, enc: str = None):
    with _engine.connect() as conn:
        if enc:
            conn.exec_driver_sql(f"SET client_encoding TO '{enc}'")
        return pd.read_sql(text(sql), conn, params=params or {})


@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)


def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    stats = db.command("dbstats")
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    return {
        "DB": db_name,
        "Collections": f"{len(colls):,}",
        "Total docs (est.)": f"{total_docs:,}",
        "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "Version": info.get("version", "unknown"),
    }


@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list, pre_hook: str | None = None):
    db = _client[db_name]

    if pre_hook == "fill_active_vehicles":
        active = db.Trip.distinct("vehicle_id", {"status": "IN_PROGRESS"})
        stages = [{"$match": {"vehicle_id": {"$in": active}}}]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()


def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")

    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(
            df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean"
        )
        st.plotly_chart(
            px.imshow(
                pivot,
                aspect="auto",
                origin="upper",
                labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"]),
            ),
            use_container_width=True,
        )
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)



st.set_page_config(page_title="SMART CAMPUS TRANSPORTATION Dashboard", layout="wide")
st.title("SMART CAMPUS TRANSPORTATION Dashboard (Postgres + MongoDB)")

with st.sidebar:
    st.header("Connections")
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"])
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    role = st.selectbox("User role", ["user", "dispatcher", "administrator", "sensor technician", "data scientist", "all"], index=5)
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    trip_id = st.number_input("Trip ID", min_value=1, value=1, step=1)
    vehicle_id = st.number_input("Vehicle ID", min_value=1, value=1, step=1)
    segment_id = st.number_input("Segment ID", min_value=1, value=1, step=1)
    days = st.slider("Last N days", 1, 90, 7)
    sensor_id = st.number_input("Sensor ID", min_value=1, value=2, step=1)
    last_hours = st.slider("Last N hours", 1, 48, 24)

    PARAMS_CTX = {
        "user_id": int(user_id),
        "trip_id": int(trip_id),
        "vehicle_id": int(vehicle_id),
        "segment_id": int(segment_id),
        "days": int(days),
        "sensor_id": int(sensor_id),
        "last_hours": int(last_hours),
    }


st.subheader("Postgres")
try:
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres query", expanded=True):
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            role_l = role.lower()

            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role_l in t

            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])
            st.code(sql, language="sql")

            run = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(
                    eng,
                    sql,
                    params=params,
                    enc=os.getenv("PG_CLIENT_ENCODING", "UTF8"),
                )
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")


if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)

        info = mongo_overview(mongo_client, mongo_db)
        st.metric(label="DB", value=info["DB"])
        st.metric(label="Collections", value=info["Collections"])
        st.metric(label="Total docs (est.)", value=info["Total docs (est.)"])
        st.metric(label="Storage", value=info["Storage"])
        st.metric(label="Version", value=info["Version"])

        with st.expander("Run Mongo aggregation / find", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]

            st.write(f"**Collection:** `{q['collection']}`")
            if "aggregate" in q:
                st.code(str(q["aggregate"]), language="python")
            if "find" in q:
                st.code(f"find({q['find']}, {q.get('projection', {})})", language="javascript")

            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                db = mongo_client[mongo_db]
                coll = q["collection"]


                if "find" in q:
                    cursor = (
                        db[coll]
                        .find(q["find"], q.get("projection", {}))
                        .sort(q.get("sort", []))
                        .limit(q.get("limit", 0))
                    )
                    docs = list(cursor)
                    dfm = pd.json_normalize(docs) if docs else pd.DataFrame()

                else:
                    stages = q["aggregate"]
                    pre_hook = q.get("pre_hook")
                    if pre_hook == "fill_active_vehicles":
                        active = db.Trip.distinct("vehicle_id", {"status": "IN_PROGRESS"})
                        stages = [{"$match": {"vehicle_id": {"$in": active}}}]
                    dfm = run_mongo_aggregate(mongo_client, mongo_db, coll, stages)

                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")
# -----------------------------------------------------------
# 作业版 app.py  2025-06（修复 datetime 未定义）
# -----------------------------------------------------------
import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

# 0. 自动感知 schema -------------------------------------------------
PG_SCHEMA = os.getenv("PG_SCHEMA", "public")
def qualify(sql: str) -> str:
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# 1. 配置区 ----------------------------------------------------------
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:541011@localhost:5432/Smart Campus Transportation"),
        "queries": {
            "User: remaining seats for trip (table)": {
                "sql": """
                    SELECT v.max_capacity - COUNT(t.seat_no) AS remaining_seats
                    FROM   {S}.Vehicle v
                    JOIN   {S}.Trip    tr ON tr.vehicle_id = v.vehicle_id
                    LEFT   JOIN {S}.Ticket t ON t.trip_id = tr.trip_id AND t.status = 'VALID'
                    WHERE  tr.trip_id = :trip_id
                    GROUP  BY v.max_capacity;
                """,
                "chart": {"type": "table"},
                "tags": ["user"],
                "params": ["trip_id"]
            },
            "Administrator: daily alerts by severity (pie)": {
                "sql": """
                    SELECT severity, COUNT(*) AS cnt
                    FROM   {S}.vehicle_alerts
                    WHERE  ts >= CURRENT_DATE
                    GROUP BY severity;
                """,
                "chart": {"type": "pie", "names": "severity", "values": "cnt"},
                "tags": ["administrator"],
                "params": []
            },
            # ====== 新增 Pg 查询：Top-5 平均余额最低的角色 =========
            "Administrator: Top-5 roles with lowest average balance (bar)": {
                "sql": """
                    SELECT role::text AS role_name,
                           COUNT(*) AS user_count,
                           ROUND(AVG(card_balance),2) AS avg_balance
                    FROM   {S}."User"
                    GROUP  BY role
                    ORDER  BY avg_balance ASC
                    LIMIT 5;
                """,
                "chart": {"type": "bar", "x": "role_name", "y": "avg_balance"},
                "tags": ["administrator"],
                "params": []
            },
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB", "smart_campus_nosql"),
        "queries": {
            # ====== 新增 3 个 Mongo 聚合 ===========================
            "Telemetry: Highest temp per vehicle (last 24h)": {
                "collection": "vehicle_telemetry",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$vehicle_id", "max_temp": {"$max": "$temp"}}},
                    {"$sort": {"max_temp": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "max_temp"}
            },
            "Telemetry: Lowest battery per vehicle (last 24h)": {
                "collection": "vehicle_telemetry",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$vehicle_id", "min_battery": {"$min": "$battery"}}},
                    {"$sort": {"min_battery": 1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "min_battery"}
            },
            "Telemetry: Average speed heat-map (last 24h)": {
                "collection": "vehicle_telemetry",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": "$vehicle_id", "avg_speed": {"$avg": "$speed"}}},
                    {"$sort": {"avg_speed": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "avg_speed"}
            },
        }
    }
}

# 2. UI / 引擎初始化函数（略，与旧版相同） ---------------------------
st.set_page_config(page_title="SMART CAMPUS TRANSPORTATION Dashboard", layout="wide")
st.title("SMART CAMPUS TRANSPORTATION Dashboard (Postgres + MongoDB)")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
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
        "Version": info.get("version", "unknown")
    }

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list):
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
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
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# 3. 侧边栏 ----------------------------------------------------------
with st.sidebar:
    st.header("Connections")
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"])
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    role = st.selectbox("User role", ["user", "dispatcher", "administrator", "all"], index=3)
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    trip_id = st.number_input("Trip ID", min_value=1, value=1, step=1)
    vehicle_id = st.number_input("Vehicle ID", min_value=1, value=1, step=1)
    segment_id = st.number_input("Segment ID", min_value=1, value=1, step=1)
    days = st.slider("Last N days", 1, 90, 7)
    sensor_id = st.number_input("Sensor ID", min_value=1, value=1, step=1)
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

# 4. Postgres 面板 ---------------------------------------------------
st.subheader("Postgres")
try:
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres query", expanded=True):
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
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
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# 5. Mongo 面板 =======================================================
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], q["aggregate"])
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")
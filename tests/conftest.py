"""PostgreSQL tests use an isolated disposable database, never Settings."""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@pytest.fixture
def db():
    url = os.environ.get("SQM_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "set SQM_TEST_DATABASE_URL for isolated PostgreSQL tests"
        )
    base = make_url(url)
    if base.get_backend_name() != "postgresql":
        pytest.fail("SQL integration tests require PostgreSQL")
    name = "sqm_test_" + uuid.uuid4().hex
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(base.set(database=name))
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA sqm"))
            conn.execute(
                text("""CREATE TABLE sqm.suppliers (
                supplier_id text PRIMARY KEY, supplier_name text,
                tier text, active boolean, onboarded_at date)""")
            )
            conn.execute(
                text("""INSERT INTO sqm.suppliers VALUES
                ('S-0417', 'Cobalt Machining', 'B', true, '2015-09-01'),
                ('S-0002', 'New Supplier', 'C', true, '2026-06-01')""")
            )
            conn.execute(
                text("""CREATE TABLE sqm.ncrs (
                ncr_id text PRIMARY KEY, supplier_id text,
                category text, severity int, cost_impact_usd numeric,
                discovered_at timestamp, closed_at timestamp)""")
            )
            rows = []
            k = 0
            for month, count in [(6, 46), (7, 46), (8, 47)]:
                severities = (
                    [5] * 7 + [4, 3] + [2] * 38
                    if month == 8
                    else [2] * count
                )
                for severity in severities:
                    k += 1
                    rows.append(
                        {
                            "id": f"NCR-2026-{k:04d}",
                            "severity": severity,
                            "timestamp": f"2026-{month:02d}-15",
                        }
                    )
            conn.execute(
                text("""INSERT INTO sqm.ncrs VALUES (
                :id, 'S-0417', 'dimensional', :severity, 100,
                CAST(:timestamp AS timestamp), NULL)"""),
                rows,
            )
            conn.execute(
                text("""CREATE TABLE sqm.audits (
                audit_id text PRIMARY KEY, supplier_id text,
                audit_date date, audit_score double precision,
                findings_count int)""")
            )
            conn.execute(
                text("""INSERT INTO sqm.audits VALUES
                ('A-1','S-0417','2026-08-01',71,3),
                ('A-2','S-0417','2026-09-01',99,0)""")
            )
            conn.execute(
                text("""CREATE TABLE sqm.cars (
                car_id text PRIMARY KEY, supplier_id text,
                opened_at timestamp, closed_at timestamp)""")
            )
            conn.execute(
                text("""INSERT INTO sqm.cars
                SELECT 'CAR-'||i, 'S-0417', TIMESTAMP '2026-07-01', NULL
                FROM generate_series(1,4) i""")
            )
            conn.execute(
                text("""CREATE TABLE sqm.purchase_orders (
                po_line_id text PRIMARY KEY, supplier_id text,
                quantity int, promised_date date)""")
            )
            conn.execute(
                text("""INSERT INTO sqm.purchase_orders
                SELECT 'PO-'||i, 'S-0417', 10, DATE '2026-08-14'
                FROM generate_series(1,100) i""")
            )
            conn.execute(
                text("""CREATE TABLE sqm.goods_receipts (
                goods_receipt_id text PRIMARY KEY, po_line_id text,
                supplier_id text, received_at timestamp, quantity int)""")
            )
            conn.execute(
                text("""INSERT INTO sqm.goods_receipts
                SELECT 'GR-'||i,'PO-'||i,'S-0417',
                CASE WHEN i<=83 THEN TIMESTAMP '2026-08-14'
                     ELSE TIMESTAMP '2026-08-15' END,10
                FROM generate_series(1,100) i""")
            )
            conn.execute(
                text("""CREATE TABLE sqm.inspections (
                inspection_id text PRIMARY KEY, supplier_id text,
                inspected_at timestamp, passed boolean)""")
            )
            conn.execute(
                text("""INSERT INTO sqm.inspections
                SELECT 'I-'||i, 'S-0417', TIMESTAMP '2026-08-15',i<=912
                FROM generate_series(1,1000) i""")
            )
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(
                text(f'DROP DATABASE "{name}" WITH (FORCE)')
            )
        admin.dispose()

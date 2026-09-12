# Chapter 5 — 5.4 Python to HANA
from sqlalchemy import URL, create_engine

from sqm_ai.settings import get_settings

s = get_settings()
engine = create_engine(
    URL.create(
        "hana+hdbcli", username=s.hana_user,
        password=s.hana_password.get_secret_value(),
        host=s.hana_host, port=s.hana_port,
    ),
    connect_args={
        "encrypt": True, "sslValidateCertificate": True,
    },
)

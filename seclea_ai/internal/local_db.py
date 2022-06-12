import datetime
import json
import traceback
from typing import List

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, synonym

Base = declarative_base()

SQLITE = "sqlite"
DATABASE = "seclea_db.sqlite"  # TODO specify where the db file is located
# Table Names
RECORD = "record"
AUTHSERVICE = "auth_service"


class Record(Base):
    __tablename__ = RECORD

    id = Column(Integer, primary_key=True, autoincrement=True)  # TODO improve.
    remote_id = Column(Integer)  # TODO this may change to string for uuids
    entity = Column(String)  # TODO remove or convert to ForeignKey - here for debugging for now.
    _dependencies = Column(String)  # this will be a list of ids
    status = Column(String)  # TODO change to enum
    timestamp = Column(DateTime)
    # only used for datasets and modelstates.
    path = Column(String)

    @property
    def dependencies(self) -> List[int]:
        return json.loads(self._dependencies)

    @dependencies.setter
    def dependencies(self, value: List[int]):
        self._dependencies = json.dumps(value)

    dependencies = synonym("_dependencies", descriptor=dependencies)


class Authservice(Base):
    __tablename__ = AUTHSERVICE

    id = Column(Integer, primary_key=True)
    key = Column(String)
    value = Column(String)


class MyDatabase:
    # http://docs.sqlalchemy.org/en/latest/core/engines.html
    DB_ENGINE = {SQLITE: "sqlite:///seclea_db"}

    # Main DB Connection Ref Obj
    db_engine = None
    session = None

    # TODO see if we can fix sec issue
    def __init__(self, dbtype=SQLITE, username="", password="", dbname=DATABASE):  # nosec
        dbtype = dbtype.lower()
        if dbtype in self.DB_ENGINE.keys():
            engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
            self.db_engine = create_engine(engine_url)
            Session = scoped_session(sessionmaker(bind=self.db_engine))
            self.session = Session()
            Record.__table__.create(bind=self.db_engine, checkfirst=True)
            Authservice.__table__.create(bind=self.db_engine, checkfirst=True)
        else:
            print("DBType is not found in DB_ENGINE")

    def create_record(self, entity: str, status: str, dependencies: List[int] = None) -> int:
        rec = Record(
            entity=entity,
            status=status,
            timestamp=datetime.datetime.now(),
            dependencies=json.dumps(dependencies) if dependencies is not None else None,
        )
        self.session.add(rec)
        self.session.commit()
        self.session.refresh(rec)
        return rec.id

    def get_record(self, record_id) -> Record:
        return self.session.get(Record, record_id)

    def update_record(self, record: Record):
        try:
            self.session.add(record)
            self.session.commit()
        except Exception:  # TODO make more specific
            self.session.rollback()
            raise Exception("DB failed to commit - see traceback")

    def set_auth_key(self, key, value):
        """
        add auth key in auth_service table and update it if already exists
        """
        try:
            auth_key = self.session.query(Authservice).filter(Authservice.key == key).first()
            if auth_key:
                auth_key.value = value
                self.session.commit()
            else:
                self.session.add(Authservice(key=key, value=value))
                self.session.commit()
        except Exception:  # TODO make more specific
            self.session.rollback()
            traceback.print_exc()
            return False
        return True

    def get_auth_key(self, key):
        """
        get auth keys from auth_service table
        """
        auth_key = self.session.query(Authservice).filter(Authservice.key == key).first()
        return auth_key.value if auth_key else False

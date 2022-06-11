import datetime
import json
import sys
import traceback
from json import JSONDecodeError
from typing import List, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

SQLITE = "sqlite"
DATABASE = "seclea_db.sqlite"
# Table Names
DATASETMODELSTATE = "dataset_modelstate"
TRAININGRUN = "training_run"
RECORD = "record"
STATUSMONITOR = "status_monitor"
AUTHSERVICE = "auth_service"


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
            Session = sessionmaker()
            Session.configure(bind=self.db_engine)
            self.session = Session()
            DatasetModelstate.__table__.create(bind=self.db_engine, checkfirst=True)
            StatusMonitor.__table__.create(bind=self.db_engine, checkfirst=True)
            Authservice.__table__.create(bind=self.db_engine, checkfirst=True)
        else:
            print("DBType is not found in DB_ENGINE")

    def save_datasetmodelstate(self, object, status):
        self.session.add(object)
        self.session.commit()
        self.add_status(object.id, status)

    def add_status(self, pid, status):
        sm = StatusMonitor(pid=pid, status=status, timestamp=datetime.datetime.now())
        self.session.add(sm)
        self.session.commit()

    def create_record(self, entity: str, status: str, dependencies: List[int] = None) -> int:
        rec = Record(
            entity=entity,
            status=status,
            timestamp=datetime.datetime.now(),
            dependencies=json.dumps(dependencies) if dependencies is not None else None,
        )
        self.session.add(rec)
        self.session.commit()
        self.session.refresh_obj(rec)
        return rec.id

    def get_record_status(self, record_id: int) -> Optional[str]:
        try:
            record = self.session.get(record_id)
            return record.status
        except JSONDecodeError:
            return None
        except Exception:
            # TODO make more specific
            traceback.print_exc()

    def set_record_status(self, record_id: int, status: str):
        try:
            record = self.session.get(record_id)
            record.status = status
            self.session.add(record)
            self.session.commit()
        except Exception:
            self.session.rollback()
            # TODO make more specific
            traceback.print_exc()

    def get_record_dependencies(self, record_id: int) -> Optional[List[int]]:
        try:
            record = self.session.get(record_id)
            return json.loads(record.dependencies)
        except JSONDecodeError:
            return None
        except Exception:
            self.session.rollback()
            # TODO make more specific
            traceback.print_exc()

    def set_record_dependencies(self, record_id: int, dependencies: List[int]) -> None:
        try:
            record = self.session.get(record_id)
            record.dependencies = json.dumps(dependencies)
            self.session.add(record)
            self.session.commit()
        except Exception:
            self.session.rollback()
            # TODO make more specific
            traceback.print_exc()

    def get_record_paths(self, record_id: int) -> Optional[List[str]]:
        try:
            record = self.session.get(record_id)
            return [record.path, record.comp_path]
        except JSONDecodeError:
            return None
        except Exception:
            self.session.rollback()
            # TODO make more specific
            traceback.print_exc()

    def set_record_paths(self, record_id: int, path: str, comp_path: str) -> None:
        try:
            record = self.session.get(record_id)
            record.path = path
            record.comp_path = comp_path
            self.session.add(record)
            self.session.commit()
        except Exception:
            # TODO make more specific
            traceback.print_exc()

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
        except Exception as e:
            self.session.rollback()
            print("Exception encountered %s" % e.with_traceback(sys.exc_info()[2]))
            return False
        return True

    def get_auth_key(self, key):
        """
        get auth keys from auth_service table
        """
        auth_key = self.session.query(Authservice).filter(Authservice.key == key).first()
        return auth_key.value if auth_key else False


class DatasetModelstate(Base):
    __tablename__ = DATASETMODELSTATE

    id = Column(Integer, primary_key=True)
    name = Column(String)
    path = Column(String)
    comp_path = Column(String)
    project = Column(String)
    organization = Column(String)


class Record(Base):
    __tablename__ = RECORD

    id = Column(Integer, primary_key=True, autoincrement=True)  # TODO improve.
    remote_id = Column(Integer)  # TODO this may change to string for uuids
    entity = Column(String)  # TODO remove or convert to ForeignKey - here for debugging for now.
    dependencies = Column(String)  # this will be a list of ids
    status = Column(String)  # TODO change to enum
    timestamp = Column(DateTime)
    # only used for datasets and modelstates.
    path = Column(String)
    comp_path = Column(String)


class StatusMonitor(Base):
    __tablename__ = STATUSMONITOR

    id = Column(Integer, primary_key=True, autoincrement=True)
    pid = Column(Integer, ForeignKey("dataset_modelstate.id"))
    status = Column(String)
    timestamp = Column(DateTime)


class Authservice(Base):
    __tablename__ = AUTHSERVICE

    id = Column(Integer, primary_key=True)
    key = Column(String)
    value = Column(String)

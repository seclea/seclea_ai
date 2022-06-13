import datetime
import json
import traceback
from enum import Enum
from typing import Dict, List

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, synonym

Base = declarative_base()

SQLITE = "sqlite"
DATABASE = "seclea_db.sqlite"  # TODO specify where the db file is located
# Table Names
RECORD = "record"
AUTHSERVICE = "auth_service"


class RecordStatus(Enum):
    IN_MEMORY = "in_memory"
    STORED = "stored"
    SENT = "sent"
    STORE_FAIL = "store_fail"
    SEND_FAIL = "send_fail"


class Record(Base):
    __tablename__ = RECORD

    id = Column(Integer, primary_key=True, autoincrement=True)  # TODO improve.
    remote_id = Column(Integer)  # TODO this may change to string for uuids
    entity = Column(String)  # TODO remove or convert to ForeignKey - here for debugging for now.
    key = Column(String)  # mainly for tracking datasets and training runs may need to remove
    _dependencies = Column(String)  # this will be a list of ids
    status = Column(String)  # TODO change to enum
    timestamp = Column(DateTime)
    # only used for datasets and modelstates.
    path = Column(String)
    # only used for datasets - probably need to factor out a lot of this.
    _dataset_metadata = Column(String)

    @property
    def dependencies(self) -> List:
        return json.loads(self._dependencies)

    @dependencies.setter
    def dependencies(self, value: List):
        self._dependencies = json.dumps(value)

    @property
    def dataset_metadata(self) -> Dict:
        return json.loads(self._dataset_metadata)

    @dataset_metadata.setter
    def dataset_metadata(self, value: Dict):
        self._dataset_metadata = json.dumps(value)

    dependencies = synonym("_dependencies", descriptor=dependencies)
    dataset_metadata = synonym("_metadata", descriptor=dataset_metadata)


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

    def create_record(
        self,
        entity: str,
        status: str,
        key: str = None,
        dependencies: List[int] = None,
        dataset_metadata: Dict = None,
    ) -> int:
        rec = Record(
            entity=entity,
            status=status,
            key=key,
            timestamp=datetime.datetime.now(),
            dependencies=dependencies,
            dataset_metadata=dataset_metadata,
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

    def search_record(self, key):
        # TODO check return value - should be None if not exist.
        return self.session.query(Record).filter(Record.key == key).first()

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

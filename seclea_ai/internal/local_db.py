import sys
import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, PrimaryKeyConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

SQLITE = 'sqlite'
DATABASE = 'seclea_db.sqlite'
# Table Names
DATASETMODELSTATE = 'dataset_modelstate'
STATUSMONITOR = 'status_monitor'
AUTHSERVICE = 'auth_service'

class MyDatabase:
    # http://docs.sqlalchemy.org/en/latest/core/engines.html
    DB_ENGINE = {
        SQLITE: 'sqlite:///seclea_db'
    }

    # Main DB Connection Ref Obj
    db_engine = None
    session = None
    def __init__(self, dbtype=SQLITE, username='', password='', dbname=DATABASE):
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
    training_run = Column(String)


class StatusMonitor(Base):
    __tablename__ = STATUSMONITOR

    id = Column(Integer, primary_key=True, autoincrement=True)
    pid = Column(Integer, ForeignKey('dataset_modelstate.id'))
    status = Column(String)
    timestamp = Column(DateTime)


class Authservice(Base):
    __tablename__ = AUTHSERVICE

    id = Column(Integer, primary_key=True)
    key = Column(String)
    value = Column(String)
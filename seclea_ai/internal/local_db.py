from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

SQLITE = 'sqlite'

# Table Names
DATASETS = 'collection_datasets'
MODELSTATES = 'collection_modelstates'


class MyDatabase:
    # http://docs.sqlalchemy.org/en/latest/core/engines.html
    DB_ENGINE = {
        SQLITE: 'sqlite:///seclea_db'
    }

    # Main DB Connection Ref Obj
    db_engine = None
    session = None
    def __init__(self, dbtype, username='', password='', dbname=''):
        dbtype = dbtype.lower()
        if dbtype in self.DB_ENGINE.keys():
            engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
            self.db_engine = create_engine(engine_url)
            Session = sessionmaker()
            Session.configure(bind=self.db_engine)
            self.session = Session()
            print(self.db_engine)
        else:
            print("DBType is not found in DB_ENGINE")

    def create_tables(self):
        metadata = MetaData()
        collection_datasets = Table(DATASETS, metadata,
                      Column('id', Integer, primary_key=True),
                      Column('name', String),
                      Column('path', String),
                      Column('comp_path', String),
                      Column('project', String),
                      Column('organization', String),
                      Column('status', String)
                      )

        collection_modelstates = Table(MODELSTATES, metadata,
                         Column('id', Integer, primary_key=True),
                         Column('path', String),
                         Column('project', String),
                         Column('organization', String),
                         Column('training_run', String),
                         Column('status', String)
                         )

        try:
            metadata.create_all(self.db_engine)
        except Exception as e:
            print("Error occurred during Table creation!")
            print(e)


class Datasets(Base):
    __tablename__ = DATASETS

    id = Column(Integer, primary_key=True)
    name = Column(String)
    path = Column(String)
    comp_path = Column(String)
    project = Column(String)
    organization = Column(String)
    status = Column(String)

class Modelstates(Base):
    __tablename__ = MODELSTATES

    id = Column(Integer, primary_key=True)
    path = Column(String)
    project = Column(String)
    organization = Column(String)
    training_run = Column(String)
    status = Column(String)
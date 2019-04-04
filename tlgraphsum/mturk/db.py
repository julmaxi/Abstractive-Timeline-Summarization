from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import ForeignKey, create_engine, Column
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.types import JSON, String, Integer, VARCHAR, TypeDecorator, Boolean

import json

class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string.

    Usage::

        JSONEncodedDict(255)

    """

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


Base = declarative_base()
Session = sessionmaker()


class AutonameBase:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()


class HITType(AutonameBase, Base):
    id = Column(String, primary_key=True)


class HIT(AutonameBase, Base):
    id = Column(String, primary_key=True)
    type_id = Column(String, ForeignKey("hittype.id"))
    topic_name = Column(String)
    timeline_name = Column(String)
    system_name = Column(String)

    assignments = relationship("HITAssignment", backref="hit")

    questions = relationship("Question", backref="hit", collection_class=ordering_list("index"))


class Question(AutonameBase, Base):
    id = Column(Integer, primary_key=True)
    text = Column(String)
    type = Column(String)
    gold_answer = Column(String)

    index = Column(Integer, nullable=False)

    hit_id = Column(String, ForeignKey("hit.id"))


class WorkerAnswer(AutonameBase, Base):
    question_id = Column(String, ForeignKey("question.id"), nullable=False, index=True, primary_key=True)
    assignment_id = Column(String, ForeignKey("hitassignment.id"), nullable=False, index=True, primary_key=True)

    text = Column(String)
    timestamp = Column(Integer)
    user_did_not_find_answer = Column(Boolean)

    question = relationship("Question", backref="worker_answers")
    assignment = relationship("HITAssignment", backref="answers")


class HITAssignment(AutonameBase, Base):
    id = Column(String, primary_key=True)
    mturk_worker_id = Column(String)
    hit_id = Column(String, ForeignKey("hit.id"))
    readability_rating = Column(Integer)

    start_time = Column(Integer)
    end_time = Column(Integer)


def open_db(db_url):
    engine = create_engine(db_url)
    Session.configure(bind=engine)

    Base.metadata.create_all(engine)

    return Session()

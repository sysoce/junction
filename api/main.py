import cherrypy
import sqlalchemy
import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, Float, DateTime
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

class Power(object):
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    power = Column(Float)
    timestamp = Column(DateTime)

    def to_dict(self):
        row = {"user_id": self.user_id, "power": self.power, "timestamp": self.timestamp.strftime(TIMESTAMP_FORMAT)}
        return row

class Production(Power, Base):
    __tablename__ = 'production'

class Consumption(Power, Base):
    __tablename__ = 'consumption'

class ProductionForecast(Power, Base):
    __tablename__ = 'production_forecast'

class ConsumptionForecast(Power, Base):
    __tablename__ = 'consumption_forecast'


class ElectricityManager(object):
	
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_past_production(self, user_id=None):
        session = Session()
        try:
            if user_id:
                past_production = session.query(Production).order_by(Production.id).filter_by(user_id=user_id).all()
            else:
                past_production = session.query(Production).order_by(Production.id).all()
            result = [element.to_dict() for element in past_production]
            return result
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_production_forecast(self, user_id=None):
        session = Session()
        try:
            if user_id:
                production_forecast = session.query(ProductionForecast).order_by(ProductionForecast.id).filter_by(user_id=user_id).all()
            else:
                production_forecast = session.query(ProductionForecast).order_by(ProductionForecast.id).all()
            result = [element.to_dict() for element in production_forecast]
            return result
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_consumption_forecast(self, user_id=None):
        session = Session()
        try:
            if user_id:
                consumption_forecast = session.query(ConsumptionForecast).order_by(ConsumptionForecast.id).filter_by(user_id=user_id).all()
            else:
                consumption_forecast = session.query(ConsumptionForecast).order_by(ConsumptionForecast.id).all()
            result = [element.to_dict() for element in consumption_forecast]
            return result
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_past_consumption(self, user_id=None):
        session = Session()
        try:
            if user_id:
                past_consumption = session.query(Consumption).order_by(Consumption.id).filter_by(user_id=user_id).all()
            else:
                past_consumption = session.query(Consumption).order_by(Consumption.id).all()
            result = [element.to_dict() for element in past_consumption]
            return result
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    @cherrypy.tools.allow(methods=['POST'])
    def insert_production(self):
        input_json = cherrypy.request.json
        # 2019-02-01T04:00:00Z
        date_time_str = input_json["timestamp"]
        date_time_obj = datetime.datetime.strptime(date_time_str, TIMESTAMP_FORMAT)
        session = Session()
        try:
            production_db_entry = Production(user_id=input_json["user_id"], power=input_json["power"], timestamp=date_time_obj)
            session.add(production_db_entry)
            session.commit()
            return {"message": "success"}
        except:
            session.rollback()
            raise
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    @cherrypy.tools.allow(methods=['POST'])
    def insert_production_forecast(self):
        input_json = cherrypy.request.json
        date_time_str = input_json["timestamp"]
        date_time_obj = datetime.datetime.strptime(date_time_str, TIMESTAMP_FORMAT)
        session = Session()
        try:
            production_forecast_db_entry = ProductionForecast(user_id=input_json["user_id"], power=input_json["power"], timestamp=date_time_obj)
            session.add(production_forecast_db_entry)
            session.commit()
            return {"message": "success"}
        except:
            session.rollback()
            raise
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    @cherrypy.tools.allow(methods=['POST'])
    def insert_consumption(self):
        input_json = cherrypy.request.json
        # 2019-02-01T04:00:00Z
        date_time_str = input_json["timestamp"]
        date_time_obj = datetime.datetime.strptime(date_time_str, TIMESTAMP_FORMAT)
        session = Session()
        try:
            consumption_db_entry = Consumption(user_id=input_json["user_id"], power=input_json["power"], timestamp=date_time_obj)
            session.add(consumption_db_entry)
            session.commit()
            return {"message": "success"}
        except:
            session.rollback()
            raise
        finally:
            session.close()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    @cherrypy.tools.allow(methods=['POST'])
    def insert_consumption_forecast(self):
        input_json = cherrypy.request.json
        # 2019-02-01T04:00:00Z
        date_time_str = input_json["timestamp"]
        date_time_obj = datetime.datetime.strptime(date_time_str, TIMESTAMP_FORMAT)
        session = Session()
        try:
            consumption_forecast_db_entry = ConsumptionForecast(user_id=input_json["user_id"], power=input_json["power"], timestamp=date_time_obj)
            session.add(consumption_forecast_db_entry)
            session.commit()
            return {"message": "success"}
        except:
            session.rollback()
            raise
        finally:
            session.close()


if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:example@postgres:5432/postgres', echo=True)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.quickstart(ElectricityManager(), '/')

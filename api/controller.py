from sqlalchemy import create_engine
import pandas as pd
import modelFile

engine = create_engine("mysql://admin@localhost/princton")
db_connection = engine.connect()

#Get last 50 records from the DB
df = pd.read_sql('SELECT * FROM princton ORDER BY Time limit 50', con=db_connection)

#Convert query result to JSON & send to the model 
out = df.to_json(orient='index') #can use split, records or columns as well
forcst = modelFile.function()

#Post to DB
df.to_sql(name='princtonForcst',con=db_connection,if_exists='append') 

con.close()

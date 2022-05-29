#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m10_model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Car_Predict',
    description='Car_Prediction_API')

ns = api.namespace('predict', 
     description='Car select Regressor')
   
parser = api.parser()

parser.add_argument(
    "Year",   
    type=int, 
    required=True, 
    help='Please add Year', 
    location='args')

parser.add_argument(
    "Mileage",   
    type=int, 
    required=True,
    help='Please add Mileage', 
    location='args')

parser.add_argument(
    "State",   
    type=str, 
    required=True, 
    help='Please add State', 
    location='args')

parser.add_argument(
    "Make",   
    type=str, 
    required=True, 
    help='Please add Make', 
    location='args')

parser.add_argument(
    "Model",   
    type=str, 
    required=True, 
    help='Please add Model', 
    location='args')


resource_fields = api.model('Resource', {
    'Year': fields.Integer,
    'Mileage': fields.Integer,
    'State': fields.String,
    'Make': fields.String,
    'Model': fields.String
})

@ns.route('/')
class CarPriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['Year'],args['Mileage'],args['State'],args['Make'],args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)

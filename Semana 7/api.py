#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from model_movies_logreg import predict_movies

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Clasificación de género de películas',
    description='API para Clasificación de género de películas')

ns = api.namespace('predict', 
     description='Movies Classifier')
   
parser = api.parser()

parser.add_argument(
    'Descripcion', 
    type=str, 
    required=True, 
    help='Descripcion to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class MoviesApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['URL'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)

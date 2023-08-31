from django.shortcuts import render
from django.views.generic import TemplateView
import joblib as joblib
import xgboost as xgb
import numpy as np
import pandas as pd
 
def home(request):
    context={}
    return render(request, 'index.html', context)

 
def analyze(request):
    context={}
    return render(request, 'analyze.html', context)
       
    

def about(request):
    context={}
    return render(request, 'about.html', context)


def result(request):
    cls = joblib.load('xgb_model.joblib')
    lis = []

    # Define the names of the expected parameters
    parameter_names = [
        'REMBECHOIRINT', 'CAT_FOND', 'COD_OPER_OPER', 'NUM_CRED', 'AGENCE',
        'COD_PRD_PRD', 'AGENT_ECO', 'CREDIT', 'REMBtotal'
    ]

    for param_name in parameter_names:
        param_value = request.GET.get(param_name, None)

        if param_value is not None:
            try:
                # Use a list comprehension to convert values to float
                float_value = float(param_value)
                lis.append(float_value)
            except ValueError:
                # Handle the case where the value cannot be converted to float
                return HttpResponse('Invalid value for ' + param_name, status=400)

    if len(lis) != len(parameter_names):
        # If not all parameters were provided, return an error response
        return HttpResponse('Some parameters are missing', status=400)

    # Convert the list to a NumPy array
    new_data = np.array([lis], dtype=float)

    # Create a pandas DataFrame with the same column names
    new_data_df = pd.DataFrame(new_data, columns=parameter_names)

    # Convert the new data to DMatrix
    new_dmatrix = xgb.DMatrix(new_data_df)

    # Make predictions on the new data
    new_predictions = cls.predict(new_dmatrix)
    pred = new_predictions[0]

    return render(request, 'result.html', {'result': pred})





def team(request):
    context={}
    return render(request, 'team.html', context)

def contacte(request):
    context={}
    return render(request, 'contact.html', context)

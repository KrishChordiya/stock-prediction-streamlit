import streamlit as st
import joblib

import torch
import torch.nn as nn

import pandas as pd


scaler = joblib.load('scaler.gz')

best_params = {'optimizer':'Adam','n_layerlstm': 2, 'hidden_state': 82, 'n_layersfc': 2, 'n_units_l0': 91, 'dropout_l0': 0.14192353580674466, 'n_units_l1': 57, 'dropout_l1': 0.12740415994127938, 'lr': 0.0071326214304045015}
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, best_params['hidden_state'], batch_first=True, num_layers=best_params['n_layerlstm'])
        self.fc = nn.Sequential(
            nn.Linear(best_params['hidden_state'], best_params['n_units_l0']),
            nn.ReLU(),
            nn.Dropout(best_params['dropout_l0']),
            nn.Linear(best_params['n_units_l0'], best_params['n_units_l1']),
            nn.ReLU(),
            nn.Dropout(best_params['dropout_l1']),
            nn.Linear(best_params['n_units_l1'], 1)
        )

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return self.fc(hn[-1])

model = MyModel()
model.load_state_dict(torch.load('lstmModel.pt', weights_only=True, map_location=torch.device('cpu')))
model.eval()


st.title("Stock Prediction Using LSTM Model")
st.subheader('Give Past 5 days data (Comma Seprated)')

sample_data = st.checkbox('Sample Data (Accor [2020-03-27 to 2020-04-02])')

open_value=''
close_value=''
high_value=''
low_value=''
volume_value=''
if sample_data:
    open_value = '23.9100,  24.1000,  25.0400,  26.5000,  29.0500'
    close_value = '22.9900,  23.8300,  25.0000,  25.0200,  26.3400'
    high_value='23.9100,  24.1000,  25.2400,  26.5000,  29.1700'
    low_value='22.9900,  23.8300,  24.9900,  24.9900,  26.3400'
    volume_value='250.0000,  37.0000, 336.0000, 415.0000,  57.0000'

open = st.text_input('Open', value=open_value)
close = st.text_input('Close', value=close_value)
high = st.text_input('High', value=high_value)
low = st.text_input('Low', value=low_value)
volume = st.text_input('Volume', value=volume_value)

btn = st.button('Predict')


if btn:
    try:
        open_values = torch.tensor([float(i) for i in open.split(',')])
        close_values = torch.tensor([float(i) for i in close.split(',')])
        high_values = torch.tensor([float(i) for i in high.split(',')])
        low_values = torch.tensor([float(i) for i in low.split(',')])
        volume_values = torch.tensor([float(i) for i in volume.split(',')])
        data= torch.concat((open_values, close_values, high_values,low_values, volume_values), axis=0).view(5, 5)
        data = torch.transpose(data, 0, 1).reshape(1, 25)
        data = torch.tensor(scaler.transform(data), dtype=torch.float).view(1, 5, 5)

        output = model(data).item()

        cdata = {'Day':[-4, -3, -2, -1, 0, 1], 'Close':[i.item() for i in close_values]+[output]}
        df = pd.DataFrame(cdata)
        # print(df)
        st.html(f'<h5 style="text-align: center">The Stock will close tommorow at {output}</h5>')
        st.line_chart(df, x='Day', y='Close')
        st.html('<p style="text-align: center">(Note: 0 represting a day before predicting day)</p>')
    except ValueError:
        st.error('Only number allowed')

    except RuntimeError:
        st.error('There should 5 input')

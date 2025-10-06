import plotly.graph_objects as go

# Data for Himachal Pradesh
himachal_pradesh_data = [
    {"Category": "Total Annual Ground Water Recharge", "Value": 1011},
    {"Category": "Annual Extractable Groundwater Resources", "Value": 1001},
    {"Category": "Current Annual Ground Water Extraction", "Value": 0.36},
    {"Category": "Stage of Ground Water Extraction", "Value": 35.48}
]

# Data for Sikkim
sikkim_data = [
    {"District": "Mangan", "Extractable Resource": 33.45},
    {"District": "Soreng", "Extractable Resource": 11.44},
    {"District": "Namchi", "Extractable Resource": 31.91},
    {"District": "Pakyong", "Extractable Resource": 45.21},
    {"District": "Gangtok", "Extractable Resource": 61.35},
    {"District": "Gyalshing", "Extractable Resource": 34.12}
]

# Create a figure for Himachal Pradesh data
fig_hp = go.Figure(data=[go.Bar(x=[d["Category"] for d in himachal_pradesh_data], y=[d["Value"] for d in himachal_pradesh_data])])
fig_hp.update_layout(title="Himachal Pradesh Ground Water Resources", xaxis_title="Category", yaxis_title="Value")
fig_hp.show()

# Create a figure for Sikkim data
fig_sikkim = go.Figure(data=[go.Bar(x=[d["District"] for d in sikkim_data], y=[d["Extractable Resource"] for d in sikkim_data])])
fig_sikkim.update_layout(title="Sikkim Ground Water Resources", xaxis_title="District", yaxis_title="Extractable Resource")
fig_sikkim.show()
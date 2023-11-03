from flask import Flask, render_template, request, jsonify
from test import predict_recommendation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    overall_rating = int(request.form['overall_rating'])
    seat_comfort = int(request.form['seat_comfort'])
    cabin_staff_service = int(request.form['cabin_staff_service'])
    food_and_beverages = int(request.form['food_and_beverages'])
    ground_service = int(request.form['ground_service'])
    value_for_money = int(request.form['value_for_money'])
    seat_type = request.form['seat_type']
    review = request.form['review']

    recommendation = predict_recommendation(overall_rating, seat_comfort, cabin_staff_service,
                                            food_and_beverages, ground_service, value_for_money, review, seat_type)

    return jsonify({'recommendation': recommendation})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)

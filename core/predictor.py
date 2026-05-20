last_x = None
last_y = None

velocity_x = 0
velocity_y = 0

def predict(x, y):

    global last_x
    global last_y

    global velocity_x
    global velocity_y

    if last_x is not None:

        velocity_x = x - last_x
        velocity_y = y - last_y

    last_x = x
    last_y = y

    predicted_x = x + velocity_x
    predicted_y = y + velocity_y

    return predicted_x, predicted_y

def lost_prediction():

    global last_x
    global last_y

    global velocity_x
    global velocity_y

    if last_x is None:

        return None, None

    predicted_x = last_x + velocity_x
    predicted_y = last_y + velocity_y

    return predicted_x, predicted_y

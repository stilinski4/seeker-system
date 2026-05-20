class TargetManager:

    def __init__(self):

        self.locked_id = None

    def select_target(self, boxes):

        if len(boxes) == 0:

            return None

        best_target = None

        biggest_area = 0

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            area = (x2 - x1) * (y2 - y1)

            if area > biggest_area:

                biggest_area = area

                best_target = box

        return best_target

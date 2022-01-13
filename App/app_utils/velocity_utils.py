import numpy as np
import pandas as pd
import math
from app_utils.filtering_utils import Filters
pd.options.mode.chained_assignment = None  # default='warn'


class VelocityUtils:
    def __init__(self):
        return None

    [staticmethod]

    def FindMidPoint(a: int, b: int) -> int:
        c = (a + b)/2
        return c

    def FindPosDiff(self, x):
        dx = x.diff()
        dx.fillna(0, inplace=True)
        return dx.astype(int, errors='ignore')

    def AddVelocity(self, df, id, fps, scale):
        mask = (df['object_id'] == id)
        sub_df = df[mask]
        x = sub_df.apply(lambda row: VelocityUtils.FindMidPoint(row["bb_left"], row["bb_right"]), axis=1)
        y = sub_df.apply(lambda row: VelocityUtils.FindMidPoint(row["bb_top"], row["bb_bottom"]), axis=1)
        dx = x.diff().fillna(0).round(2)
        dy = y.diff().fillna(0).round(2)
        d = np.sqrt(dx * dx + dy * dy).round(2)
        df.loc[mask, 'center_x'] = x.round(2)
        df.loc[mask, 'center_y'] = y.round(2)
        df.loc[mask, 'dx'] = dx.round(2)
        df.loc[mask, 'dy'] = dy.round(2)
        df.loc[mask, 'd'] = d.round(2)
        vel = round(d * fps * scale, 1)
        vel[vel < 1] = 0
        if(len(vel) > 10):
            filt_vel = Filters.ButterLowpass(vel, fps/60, fps, 1)
        else:
            filt_vel = vel

        filt_vel[filt_vel < 1] = 0

        df.loc[mask, 'vel'] = vel
        df.loc[mask, 'filt_vel'] = filt_vel.round(1)

        df.replace(np.nan, 0)
        return df

    def CalculateScaleForCategory(self, df, category, metric_conversion, use_width_bbox=True):

        scale = -1
        df = df[(df['category'] == category)]
        if (len(df) > 0):
            df["width"] = df.apply(lambda row: abs(row['bb_left'] - row['bb_right']), axis=1)
            df["height"] = df.apply(lambda row: abs(row['bb_bottom'] - row['bb_top']), axis=1)

            if(use_width_bbox):
                scale = metric_conversion / df["width"].mean()
            else:  # using height
                scale = metric_conversion / df["height"].mean()

        return scale

    def CalculateScale(self, df):
        # Finding the scale factor to map pixel movement to meters:
        scale = 0

        person_scale = self.CalculateScaleForCategory(df, "person", 0.5, True)
        print(f"Person scale:  {person_scale}")

        car_scale = self.CalculateScaleForCategory(df, "car", 2, True)
        print(f"Car scale:  {car_scale}")

        if(person_scale > 0):
            scale = person_scale
        elif (car_scale > 0):
            scale = car_scale

        return scale

    def CalculateScaleCameraProperites(self, camera_tilt_angle_deg, cam_height, image_height, cam_fov_deg, cam_focal_length=-1, vert_image_dim=-1):
        # Video data is captured from static camera at top position using geometric equation to calculate:
        # - perpendicular view (P)
        # - Distance between object and camera (D) based on camera height (H), field of view angle of the camera (Tc), angle of the camera (Tv)
        # Perpendicular view is used to calibrate estimated speed to standard unit

        Tv = 90 - abs(camera_tilt_angle_deg)  # in camera_tilt_angle_deg, camera down is negative
        v = vert_image_dim  # vertical dimension of 35 mm image format which can be found fromcamera specifications.
        f = cam_focal_length  # focal length of the camera
        H = abs(cam_height)  # height of camera above object

        if((f > 0) and (v > 0)):
            Tc = abs(2 * math.degrees(math.atan(v/(2*f))))  # field of view angle of the camera (deg)
        else:
            Tc = abs(cam_fov_deg)  # field of view angle of the camera  (deg)

        T = Tv + Tc/2
        D = round(H * math.tan(math.radians(T)), 3)  # distance between object and camera
        P = round(2 * math.tan(math.radians(Tc/2)) * math.sqrt(H*H + D*D), 3)   # perpendicular view
        k = round(P / image_height, 3)  # calibration coefficient based on perpendicular viewP and image height

        print(f"Tv: {Tv},  f: {f},  v: {v},  I_height: {image_height},   H: {H},  Tc: {Tc},  T: {T},  P: {P},  D: {D},   k: {k}")
        return k

import pandas as pd
import numpy as np

class VelocityUtils:
    def __init__(self):
        return None

    def find_mid(self, a, b):
        c = (a.astype(float) + b.astype(float))/2
        return c

    def find_delta(self, x):
        dx = x.diff()
        dx[0] = np.nan
        dx.fillna(method='backfill', inplace=True)
        return dx

    def find_vel(self, vx, vy):
        vel = np.sqrt(vx * vx + vy * vy).round(2)
        return vel#, vx, vy

    def add_vel_to_df(self, df, fps, scale):
        for id in df['object_id'].unique():
            mask = (df['object_id']==id)
            sub_df = df[mask]
            x  = self.find_mid(sub_df['bb_top'], sub_df['bb_bottom'])
            y  = self.find_mid(sub_df['bb_left'], sub_df['bb_right'])
            dx = self.find_delta(x)
            dy = self.find_delta(y)
            vx = (dx * fps * scale)
            vy = (dy * fps * scale)

            df.loc[mask, 'x']  = x
            df.loc[mask, 'y']  = y
            df.loc[mask, 'dx'] = dx
            df.loc[mask, 'dy'] = dy
            df.loc[mask, 'vx'] = (dx * fps * scale)
            df.loc[mask, 'vy'] = (dy * fps * scale)
            df.loc[mask, 'vel']  = np.sqrt(vx * vx + vy * vy).round(2)

        return df
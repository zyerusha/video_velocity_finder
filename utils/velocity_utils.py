import numpy as np


class VelocityUtils:
    def __init__(self):
        return None

    [staticmethod]

    def FindMidPoint(a: int, b: int) -> int:
        c = (a + b)/2
        return c

    def FindPosDiff(self, x):
        dx = x.diff()
        dx[0] = np.nan
        dx.fillna(method='backfill', inplace=True)
        return dx.astype(int, errors='ignore')

    def FindVel(self, vx, vy):
        vel = np.sqrt(vx * vx + vy * vy).round(2)
        return vel  # , vx, vy

    def AddVelocity(self, df, id, fps, scale, column_names):
        mask = (df['object_id'] == id)
        sub_df = df[mask]
        y = VelocityUtils.FindMidPoint(
            sub_df[column_names[3]], sub_df[column_names[1]])
        x = VelocityUtils.FindMidPoint(
            sub_df[column_names[2]], sub_df[column_names[0]])
        dx = self.FindPosDiff(x)
        dy = self.FindPosDiff(y)
        vx = (dx * fps * scale)
        vy = (dy * fps * scale)
        df.loc[mask, 'x'] = x
        df.loc[mask, 'y'] = y
        df.loc[mask, 'dx'] = dx
        df.loc[mask, 'dy'] = dy
        df.loc[mask, 'vx'] = vx
        df.loc[mask, 'vy'] = vy
        df.loc[mask, 'vel'] = self.FindVel(vx, vy)
        return df

    # def FilterVelocity(self, df, id, fps):
    #     mask = (df['object_id']==id)
    #     sub_df = df[mask]
    #     vel = sub_df.loc[mask, 'vel']
    #     df.loc[mask, 'vel_filt']  = Filters.ButterLowpass(vel, 0.5, fps, 1)
    #     return df

import numpy as np 


class SimpleLinearRegresion:
    def __init__(self, x, y):
        self.x = x 
        self.y = y 
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)


    def _calc_theta(self):
        theta_1 = sum([(i - self.x_mean) * (j - self.y_mean) for i, j in zip(self.x, self.y)]) / sum([(i - self.x_mean) ** 2 for i, j in zip(self.x, self.y)])
        theta_0 = self.y_mean - theta_1 * self.x_mean

        return theta_0, theta_1

    
    def predict(self):
        theta_0, theta_1 = self._calc_theta()
        self.y_pred = [theta_0 + theta_1 * i for i in self.x]

        return np.array(self.y_pred).reshape((len(self.y), 1))


    # root mean square error
    def rmse(self):
        return (sum([(j - i ) ** 2 for i, j in zip(self.y, self.y_pred)]) / len(self.x)) ** 0.5

    
    def r_square(self):
        # rss --> sum of squares of residuals
        # tss --> total sum of squares
        rss = sum([(i - j) ** 2 for i, j in zip(self.y, self.y_pred)])
        tss = sum([(i - self.y_mean) ** 2 for i in self.y])

        return f'{(1 - rss/tss) * 100} %'





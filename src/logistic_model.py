import numpy as np

class LogisticRegressionScratch:

    def __init__(self,learning_rate: float,epochs: int):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.theta=None
    
    def add_bias(self,X):

        ones_col = np.ones((X.shape[0], 1))
        X_bias = np.hstack((ones_col, X))
        
        return X_bias
    
    def sigmoid(self,z):

        return 1/(1+np.exp(-z))

    def forward(self,X):
        X_bias= self.add_bias(X)
        y_pred= X_bias @ self.theta
        return self.sigmoid(y_pred)
    
    def compute_loss(self,y,y_pred):

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss=-np.mean(y*np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return loss
    
    def backward(self,X,y,y_pred):
        m=len(y)
        X_bias=self.add_bias(X)
        grad_theta=(X_bias.T @ (y_pred - y)) / m

        return grad_theta
    
    def fit(self, X, y, X_test=None, y_test=None):
        self.losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        X_bias=self.add_bias(X)
        num_columns = X_bias.shape[1]
        self.theta=np.zeros((num_columns,))

        for i in range(self.epochs):
            y_pred=self.forward(X)
            loss=self.compute_loss(y,y_pred)
            grad_theta=self.backward(X,y,y_pred)
            self.theta-= self.learning_rate*grad_theta

            self.losses.append(loss)
    
            train_preds = self.predict(X)
            train_acc = np.mean(train_preds == y)
            self.train_accuracies.append(train_acc)
    
            if X_test is not None and y_test is not None:
                test_preds = self.predict(X_test)
                test_acc = np.mean(test_preds == y_test)
                self.test_accuracies.append(test_acc)            
            
            if i % 100 == 0:
                msg = f"Epoch {i} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f}"
                if X_test is not None:
                    msg += f" | Test Acc: {test_acc:.4f}"
                print(msg)

    def predict_proba(self,X_test):
        return self.forward(X_test)
    
    def predict(self,X):
        predictions = np.where( self.predict_proba(X)> 0.5, 1, 0)
        return predictions
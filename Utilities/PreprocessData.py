from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from matplotlib import pyplot as plt


class PrepareData:
    """Helper class to preprocess the data"""

    def __init__(self, df) -> None:

        self.dfrm = df
        self.LE = LabelEncoder()
        self.ONE = OneHotEncoder(drop="first", dtype=int, sparse=False)
        self.SCALER = StandardScaler()

        # Columns where Label Encoding will be applied
        self.binary_columns = [feat for feat in self.dfrm.columns if self.dfrm[feat].nunique(
        ) == 2 and str(self.dfrm[feat].dtype) != "int64"]
        # Columns where One-Hot Encoding will be applied
        self.one_hot_cols = [feat for feat in self.dfrm.columns if self.dfrm[feat].nunique(
        ) > 2 and str(self.dfrm[feat].dtype) == 'object']
        # Columns where Scaling will be applied
        self.scale_cols = [feat for feat in self.dfrm.columns if str(
            self.dfrm[feat].dtype) == "float64"]

    def apply_preprocessing(self) -> None:
        """This method applies all the preprocessing used during the training"""
        #-------------------------------------------------------#
        # TODO: Apply label encoding
        for i in self.binary_columns:
            self.dfrm[i] = self.LE.fit_transform(self.dfrm[i])
        print("✔ Label Encoding Successful!")
        #-------------------------------------------------------#
        # TODO: Apply One-Hot Encoding
        # Select the columns to encode and slice it
        to_enc = self.dfrm[self.one_hot_cols]
        # Fit and transform the data and save it
        self.ONE.fit(to_enc)
        encoded = self.ONE.transform(to_enc)
        # get the name of the column
        cols = [name[3:] for name in list(self.ONE.get_feature_names())]
        # add the feature name and it's subsequent values to the dataframe
        self.dfrm[cols] = encoded
        # drop the original columns and return the new dataframe
        self.dfrm.drop(self.one_hot_cols, axis=1, inplace=True)
        print("✔ One-Hot Encoding Successful!")
        #--------------------------------------------------------#
        # TODO : Apply Scaling to features
        to_scale = self.dfrm[self.scale_cols]
        scaled = self.SCALER.fit_transform(to_scale)
        self.dfrm[self.scale_cols] = scaled
        print("✔ Scaling Successful!")
        # return the pre-processed data
        return self.dfrm

    def __repr__(self) -> str:
        return f"""
        Applying LabelEncoding To : {self.binary_columns}
        Applying OneHotEncoding To : {self.one_hot_cols}
        Applying Scaling To : {self.scale_cols}
        """


# Helper function for plotting
def plot_graphs(clf, x, y):
    """
    Function plots the confusion matrix and roc curve
    It takes 4 parameters:

    clf : The model which is used
    X : Set of features
    y : true values
    name(optional) : The name of the graph to be displayed 
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].set_title(f"Confusion Matrix")
    ax[1].set_title("ROC/AUC")
    plot_confusion_matrix(clf, x, y, colorbar=False, ax=ax[0])
    plot_roc_curve(clf, x, y, ax=ax[1])

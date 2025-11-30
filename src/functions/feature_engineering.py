import pandas as pd

def feature_engineering(df):
    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"]+1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Ticket group size
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    if "Ticket" in df.columns:
        df = df.drop(columns=["Ticket"])
    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

    # Age band
    df["AgeBand"] = pd.cut(df["Age"], bins=[0,12,18,35,60,100], labels=False)

    return df
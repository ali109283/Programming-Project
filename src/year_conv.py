# Little function to help adding N years go when initializing the object instead of an exact date
# We only need 1 library datetime


from datetime import datetime, timedelta

def years(N: int) -> str:

    """
    The fucntion will eturns a string date representing N years ago from today. Parameters
    -------
    str: Date string in 'YYYY-MM-DD' format.
    """
    today = datetime.today()
    # Approximate 1 year as 365 days for simplicity
    start_date = today - timedelta(days=N * 365)
    return start_date.strftime("%Y-%m-%d")
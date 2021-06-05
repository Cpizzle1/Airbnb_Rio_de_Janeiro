import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from clean2 import clean_df


def two_sample_t_test(df, col):
    booked_contact_series = df[df.col ==1]
    unbooked_contact_series = df[df.ts_booking_at ==0]

    np_booked_contact = booked_contact_series.m_interactions.to_numpy()
    np_unbooked_contact= unbooked_contact_series.m_interactions.to_numpy()

    print(stats.ttest_ind(a= np_booked_contact,
                b= np_unbooked_contact,
                equal_var=False))


if __name__ == "__main__":

    listings_df =pd.read_csv('~/Downloads/2018 DA Take Home Challenge/listings.csv')
    contacts_df =pd.read_csv('~/Downloads/2018 DA Take Home Challenge/contacts.csv')
    users_df = pd.read_csv('~/Downloads/2018 DA Take Home Challenge/users.csv')

    combined_df =contacts_df.merge(listings_df, left_on ='id_listing_anon', right_on='id_listing_anon')
    combined_df2 =combined_df.merge(users_df, left_on ='id_guest_anon', right_on='id_user_anon')
    combined_df3 = combined_df2.copy()

    contact_me_df3,instant_book_df3,book_it_df3 ,combined_df3 =  clean_df(combined_df3)

    booked_contact_series = contact_me_df3[contact_me_df3.ts_booking_at ==1]
    unbooked_contact_series = contact_me_df3[contact_me_df3.ts_booking_at ==0]

    booked_contact_series_response = booked_contact_series.booked_contact_series]
    unbooked_contact_series_response

    response_time_hours

    # np_booked_contact = booked_contact_series.m_interactions.to_numpy()
    # np_unbooked_contact= unbooked_contact_series.m_interactions.to_numpy()

    # np_booked_contact_response = booked_contact_series.m_interactions.to_numpy()
    # np_unbooked_contact_response= unbooked_contact_series.m_interactions.to_numpy()

    print(stats.ttest_ind(a= np_booked_contact,
                b= np_unbooked_contact,
                equal_var=False))
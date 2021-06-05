import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sbn
from datetime import date
import scipy.stats as stats
import math

def clean_df(combined_df3):
    combined_df3.ts_booking_at = combined_df3.ts_booking_at.fillna(0)
    combined_df3.ts_booking_at = combined_df3.ts_booking_at.apply(lambda x: 0 if x==0 else 1)
    combined_df3["m_guests"] = combined_df3["m_guests"].fillna(2)
    combined_df3['booked'] = combined_df3.ts_booking_at.apply(lambda x: True if x == 1 else False)

    combined_df3['ts_interaction_first'] = pd.to_datetime(combined_df3['ts_interaction_first'],infer_datetime_format=True)
    combined_df3['ds_checkin_first'] = pd.to_datetime(combined_df3['ds_checkin_first'],infer_datetime_format=True)

    combined_df3['ds_checkout_first'] = pd.to_datetime(combined_df3['ds_checkout_first'],infer_datetime_format=True)

    combined_df3['date_interaction_first'] = pd.to_datetime(combined_df3['ts_interaction_first'].dt.date)
    combined_df3['delta_days'] = combined_df3.ds_checkin_first - combined_df3.date_interaction_first
    combined_df3['delta_days'] = pd.to_numeric(combined_df3['delta_days'].dt.days, downcast='integer')
    combined_df3['booked'] = combined_df3.ts_booking_at.apply(lambda x: True if x == 1 else False)
    d1 =  {
        '-unknown-':'new'}
    combined_df3['guest_user_stage_first'].replace(d1, inplace= True)

    combined_df3['length_of_stay'] = combined_df3['ds_checkout_first']-combined_df3['ds_checkin_first']
    combined_df3['length_of_stay_days'] = combined_df3['length_of_stay'] / np.timedelta64(1, 'D')
    combined_df3['ts_reply_at_first'] = pd.to_datetime(combined_df3['ts_reply_at_first'],infer_datetime_format=True)
    combined_df3['response_time'] = combined_df3['ts_reply_at_first']-combined_df3['ts_interaction_first']
    combined_df3['response_time_hours'] = combined_df3['response_time'] / np.timedelta64(1, 'h')

    contact_me_df3= combined_df3[combined_df3['contact_channel_first']=='contact_me']
    instant_book_df3= combined_df3[combined_df3['contact_channel_first']=='instant_book']
    book_it_df3= combined_df3[combined_df3['contact_channel_first']=='book_it']

    return contact_me_df3, instant_book_df3, book_it_df3, combined_df3




def KDE_plot_maker(df, lst ):
    kde_data_book_it_df3 = book_it_df3[lst]

    kde_cols = kde_data_book_it_df3.iloc[:,:8].columns.to_list()
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15,30))

    for col, ax in zip(kde_cols[:8], axs.flatten()):
        
        booked = kde_data_book_it_df3[kde_data_book_it_df3.ts_booking_at == 1]
        unbooked = kde_data_book_it_df3[kde_data_book_it_df3.ts_booking_at == 0]

        sbn.kdeplot(booked[col], fill=True, bw_method=0.2, color='#000080', label='Booked reservations', ax=ax)
        sbn.kdeplot(unbooked[col], fill=True, bw_method=0.2, color='#FF0000', label='Not booked reservations', ax=ax)
        ax.set_xlabel('')
        ax.set_title(col.replace('_', ' ').title())
    #     ax.set_title(col.replace('M', '').title())
        _ = ax.legend(loc='upper center')
        
    plt.suptitle("KDE Plots for Numerical Predictors Book it type",y=0.91, fontsize=25)
    plt.show()

def catagorical_stacked_graph_maker(combined_df3, lst2):
    cat_data = combined_df3[lst2]
    cat_cols = cat_data.iloc[:,:4].columns.to_list()
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15,30))

# date_format = mdates.DateFormatter('%B-%d')

    for col, ax in zip(cat_cols[:4], axs.flatten()):
        group_data = cat_data.groupby([col,'booked']).size().unstack()
        group_data.columns = ['Unbooked', 'Booked']
        group_data.plot.bar(stacked=True, ax=ax, color=['#000080', '#FF0000'], alpha = 0.5)
        if col == 'signup_date':
            ax.xaxis.set_major_formatter(date_format)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode="anchor")
        else:
            ax.tick_params(labelrotation=0)
        ax.set_xlabel("")
        ax.set_title(col.replace('_', ' ').title())
        _ = ax.legend(loc='best')

    plt.suptitle("Stacked Barcharts for Categorical Predictors",y=0.91, fontsize=25)
    plt.show() 



if __name__ == "__main__":

    listings_df =pd.read_csv('~/Downloads/2018 DA Take Home Challenge/listings.csv')
    contacts_df =pd.read_csv('~/Downloads/2018 DA Take Home Challenge/contacts.csv')
    users_df = pd.read_csv('~/Downloads/2018 DA Take Home Challenge/users.csv')

    combined_df =contacts_df.merge(listings_df, left_on ='id_listing_anon', right_on='id_listing_anon')
    combined_df2 =combined_df.merge(users_df, left_on ='id_guest_anon', right_on='id_user_anon')
    combined_df3 = combined_df2.copy()

    contact_me_df3,instant_book_df3,book_it_df3 ,combined_df3 =  clean_df(combined_df3)

    lst = ['m_guests', 'm_interactions', 'm_first_message_length_in_characters', 'total_reviews',
       'words_in_user_profile','delta_days' ,'length_of_stay_days','response_time_hours', 'ts_booking_at']
    
    KDE_plot_maker(combined_df3, lst)

    # print(contact_me_df3,instant_book_df3,book_it_df3 ,combined_df3)
    lst2 = ['guest_user_stage_first', 'room_type', 'listing_neighborhood','country', 'booked']
    catagorical_stacked_graph_maker(combined_df3, lst2)

    col_list = ['id_guest_anon', 'id_host_anon', 'id_listing_anon',
       'ts_interaction_first', 'ts_reply_at_first', 'ts_accepted_at_first','ds_checkin_first', 'ds_checkout_first','id_user_anon',
            'country','booked', 'date_interaction_first', 'response_time','listing_neighborhood']
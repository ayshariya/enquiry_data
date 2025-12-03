import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

@st.cache_resource
def load():
    model=joblib.load(r"C:\Users\Aysha Riya\OneDrive\Documents\CyberSquare_EnquiryDatas\Models\ED_rf_withnull_model.pkl")
    return model

@st.cache_data
def load_data():
    data=pd.read_csv(r"C:\Users\Aysha Riya\OneDrive\Documents\CyberSquare_EnquiryDatas\Datas\EnquiryDatas.csv")
    return data

model=load()
data=load_data()

#st.set_page_config(page_title="CyberSquare Enquiry Data Anlysis",layout='wide')
st.title('CyberSquare Enquiry Data Anlysis Dashbord')

menu=st.sidebar.radio("Navigation",["📂 Dataset Explorer", "📊 Data Visualization", "🤖 Prediction Model"])

if menu=="📂 Dataset Explorer":
    st.header("📂 Explore Dataset")
    st.subheader("🔎 Preview")
    st.dataframe(data.head(20))
    st.subheader("📊 Summary Stats")
    st.write(data.describe())
    st.subheader('columns and Datatypes')
    st.write(data.dtypes)
    st.subheader("Filter By Year")
    if 'Year_Of_Enq'in data.columns:
        year=st.selectbox("Filter By Year",options=sorted(data['Year_Of_Enq'].dropna().unique()))
        year_data=data[data["Year_Of_Enq"]==year]
        st.write(f"data of year {year}")
        st.dataframe(year_data)
elif menu=='📊 Data Visualization':
    st.header('📊Visualaizations and insights')
    cat=st.selectbox("Choose visualaization Catagory",['Year Base','Location Base','Find_Us Base','Qualification Base','Issue Faced','Looking_for based','Technology Based','Mod of study','Time Slot','Attended Staff','Revenue Details','duration'])
    if cat=='Year Base':
     st.subheader("Total Number Of Enquiries by year")
     df=data[data['Status']=='joined'].groupby(data['Year_Of_Enq']).size()
     fig=plt.figure(figsize=(13,6))
     plt.plot(df.index,df.values,color='green',linewidth=2.5,marker='o')
     plt.fill_between(
     df.index,
     df.values,
     color='lightgreen',
     alpha=0.5
     )
     plt.xlabel('Year of Enquiry')
     plt.ylabel('Count of Joined Students')
     plt.grid(linestyle="--")
     plt.show()
     st.pyplot(fig)
     st.markdown(
     """
     ### 📊 Insights:

     * <p style="color:green;">✅ Peak enrollment happened in 2021, with nearly 70% of enquiries converting to joined students.</p>

     * <p style="color:red;">📉 A downward trend continued through 2022–2023, where the conversion rate dropped sharply to about 39% — possibly due to external factors (e.g., post-pandemic uncertainty, changes in course demand, or competition).</p>

     * <p style="color:green;">📈 Partial recovery in 2024 indicates some improvement in student engagement or marketing efforts.</p>

     * <p style="color:red;">⚠️ 2025 shows a small decline again, suggesting the recovery isn’t fully sustained — there may still be some issues in conversion or student interest.</p>

     * <p style="color:red;">📉 Overall trend (2021 → 2025) is downward, with a brief rebound in 2024.</p>
     """,
     unsafe_allow_html=True
     )
     st.subheader("Year Wise Enrolled Analysis")
     df3=pd.read_excel(r"C:\Users\Aysha Riya\OneDrive\Documents\CyberSquare_EnquiryDatas\Datas\year&month.xlsx")
     df2021=df3.iloc[:11]
     fig2 = go.Figure()
     fig2.add_trace(go.Scatter(
        x=df2021['Month_Of_Enq'],
        y=df2021['Count'],
        mode='lines+markers',
        line=dict(color='green', width=2.5),
        fill='tozeroy',  # same as plt.fill_between
        fillcolor='lightgreen',
        marker=dict(size=8)
     ))

# Customize layout
     fig2.update_layout(
        title='Year 2021 Monthly Enrolled Count',
        xaxis=dict(
          title='Month',
          tickmode='array',
          tickvals=df2021['Month_Of_Enq'],
          ticktext=['Jan','Feb','Mar','Apr','May','Jun','Aug','Sep','Oct','Nov','Dec']
        ),
        yaxis_title='Number Of Joined',
        plot_bgcolor='white',
        hovermode='x unified',
        width=1000,
        height=400
     )

# Dsplay in Streamlit
     st.plotly_chart(fig2, use_container_width=True)
     st.markdown(
     """
     ### 📊 Insights:
     * <p style="color:green;">✅ Peak enrollment happened in March 2021.</p>
     * <p style="color:red;">📉 Data for July is missing — maybe no enrollments or data not recorded.</p>
     * <p style="color:green;">📈 After April, enrollment trend declined but regained after August.</p>
     * <p style="color:blue;">🔹 March is when Kerala University's academic year ends — possibly why enrollments spiked in March.</p>
     """,
     unsafe_allow_html=True
     )
     df2022=df3.iloc[11:22]
     fig3 = go.Figure()
     fig3.add_trace(go.Scatter(
        x=df2022['Month_Of_Enq'],
        y=df2022['Count'],
        mode='lines+markers',
        line=dict(color='green', width=2.5),
        fill='tozeroy',  # same as plt.fill_between
        fillcolor='lightgreen',
        marker=dict(size=8)
     ))
     fig3.update_layout(
        title='2022 Monthly Analysis',
        xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=df2022['Month_Of_Enq'],
            ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']
        ),
        yaxis_title='Count',
        plot_bgcolor='white',
        hovermode='x unified',
        width=1000,
        height=400
     )                  
     st.plotly_chart(fig3, use_container_width=True)
     st.markdown(
     """
     ### Insights:
     * <p style="color:green"> peak enrollment happend in september month in 2022 </p>
     * <p style="color:green"> enrollment trend upwards after march month  </p>
     * <p style="color:green"> Overall 2022 was good year for company</p>
     * <p style="color:blue">College closures drive enrollment increases.</p>
     """,
     unsafe_allow_html=True
     )
     df2023=df3.iloc[22:27]
     fig4 = go.Figure()
     fig4.add_trace(go.Scatter(
        x=df2023['Month_Of_Enq'],
        y=df2023['Count'],
        mode='lines+markers',
        line=dict(color='red', width=2.5),
        fill='tozeroy',  # same as plt.fill_between
        fillcolor='lightcoral',
        marker=dict(size=8)
     ))
     fig4.update_layout(
          title='2023 Monthly Analysis',
          xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=df2023['Month_Of_Enq'],
            ticktext=['Aug','Sep','Oct','Nov','Dec']
          ),
          yaxis_title='Count',
          plot_bgcolor='white',
          hovermode='x unified',
          width=1000,
          height=400
     )
     st.plotly_chart(fig4, use_container_width=True)
     st.markdown(
     """
     ### Insights:
     * <p style="color:red"> Details of 4 month only avaialble in 2023</p>
     * <p style="color:red"> only 6th enrollments happened in aug month and that is the peakest enrollment happened in 2023</p>
     * <p style="color:red"> Overall 2023 was not a good year for company</p>
     """,
     unsafe_allow_html=True
     )
     df2024=df3.iloc[27:35]
     fig5 = go.Figure()
     fig5.add_trace(go.Scatter(
        x=df2024['Month_Of_Enq'],
        y=df2024['Count'],
        mode='lines+markers',
        line=dict(color='green', width=2.5),
        fill='tozeroy',  # same as plt.fill_between
        fillcolor='lightgreen',
        marker=dict(size=8)
     ))
     fig5.update_layout(
        title='2024 Monthly Analysis',
        xaxis=dict(
          title='Month',
          tickmode='array',
          tickvals=df2024['Month_Of_Enq'],
          ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']
        ),
        yaxis_title='Count',
        plot_bgcolor='white',
        hovermode='x unified',
        width=1000,
        height=400
     )
     st.plotly_chart(fig5, use_container_width=True)
     st.markdown(
     """
     ### Insights:
     * <p style="color:green"> peak enrollment happend in may,jun,jul</p>
     * <p style="color:green"> enrollment trend upwards after april month and downward after july maintaning a pattern  </p>
     * <p style="color:green"> overall good year for company</p>
     * <p style="color:blue">College closures drive enrollment increases.</p>
     """,
     unsafe_allow_html=True
     )
     df2025=df3.iloc[35:]
     fig6=go.Figure()
     fig6.add_trace(go.Scatter(
        x=df2025['Month_Of_Enq'],
        y=df2025['Count'],
        mode='lines+markers',
        line=dict(color='red', width=2.5),
        fill='tozeroy',  # same as plt.fill_between
        fillcolor='lightcoral',
        marker=dict(size=8)
     ))
     fig6.update_layout(
        title='2025 Monthly Analysis',
        xaxis=dict(
          title='Month',
          tickmode='array',
          tickvals=df2025['Month_Of_Enq'],
          ticktext=['Jan','Feb','Mar','Apr','May','Jun']
          ),
          yaxis_title='Count',
          plot_bgcolor='white',
          hovermode='x unified',
          width=1000,
          height=400
     )
    
    
     st.plotly_chart(fig6, use_container_width=True)     
     st.markdown(
     """
     ### Insights:
     * <p style="color:blue"> only datas upto june month availabe</p>
     * <p style="color:green">upward trend till may</p>
     * <p style="color:green"> may have peakest enrollment</p>
     * <p style="color:blue">College closures drive enrollment increases.</p>
     """ ,
     unsafe_allow_html=True
     )
     st.subheader('Overall')
     status_counts = data['Status'].value_counts()

# Create bar chart
     fig7 = go.Figure(
     data=[go.Bar(
        x=status_counts.index,  # actual values, not 'Status' string
        y=status_counts.values, # counts
        text=status_counts.values,
        textposition='auto'
    )]
)

# Update layout
     fig7.update_layout(
       title='Overall Joined Status',
       xaxis_title='Status',
       yaxis_title='Count',
       template='plotly_white'
     )

# Display in Streamlit
     st.plotly_chart(fig7, use_container_width=True)
     st.markdown(
     """
     ### Insights:
     * <p style='color:green'>52% of enquries are converted to joined,total 237 joins</p>
     * <p style='color:red'>47% are cancelled,total 215</p>
     * <p style='color:blue'>Since 2025 is not over may varies at end of year,how ever over all 'average' perfomance by anlysing data of 4 years</p>
     """,
     unsafe_allow_html=True
     )
     data['Remarks']=data['Remarks']
     data['Remarks']=data['Remarks'].replace({
     'call not picking':'np',
     'closed lost':'lost',
    'no contacts':'np',
    'not applicable':'other',
    'not picking':'np',
    'not intrested':'not interested',
    'no update':'not interested',
    'no':'not interested',
    'suspected':'suspected-rivals',
    'went to abudhabi':'not interested',
    'canceled':'cancelled',
    'not responding':'not interested',
    'number busy':'np',
    'suspected softronix':'suspected-rivals',
    'waiting for interview result':'other',
    'nil':'not interested',
    'rush shedule':'schedule issue',
    'disonnected':'np',
    'not looking it':'not interested',
    'having backpapers':'other',
    'looking for ui/ux':'other',
    'no certificate needed':'not intrested',
    'joined at kondotty sql':'not interested',
    'will update':'not interested',
    'not joined':'not interested',
    'fake':'suspected-rivals',
    'found other':'not interested',
    'pg':'higher study',
    'jined':'not interested',
    'digital marketing':'not interested',
    'forwarded to arabic':'not interested',
    'going for higher studies':'higher study',
    'enquiry for friend':'other',
    'joined another institution':'not interested',
    
    
     })
     st.subheader('Reason for not joining')
     cancelled_counts = (
        data[data['Status'] == 'no']
        .groupby('Remarks')
        .size()
        .reset_index(name='Count')
     )
     fig8 = px.bar(
        cancelled_counts,
        x='Remarks',
        y='Count',
        color_discrete_sequence=['indianred'],
        title='Cancelled Students by Remark Category'
     )
     fig8.update_layout(
        xaxis_title='Remark Category',
        yaxis_title='Count',
        plot_bgcolor='white',
        hovermode='x',
        width=1000,
        height=500
     )
     fig8.update_xaxes(tickangle=90)
     st.plotly_chart(fig8, use_container_width=True)
     st.markdown(
    """
    ### Insights
    * <p style="color:red"> 79 enquiries lost becuase of call not picking</p> 
    * <p style="color:red"> 48 of enquiries have no intrest in joining</p>
    * <p style="color:red"> 15 rivals enquiry found</p>
    """,
    unsafe_allow_html=True
     )
    elif cat=='Location Base':
     st.subheader('Top5 Locations with enquiries')
     top5_locations = (
            data['Location']
        .value_counts()
        .head()
        .reset_index()
     )
     top5_locations.columns = ['Location', 'Count']

    # Create Plotly bar chart
     fig9 = px.bar(
        top5_locations,
        x='Location',
        y='Count',
        color_discrete_sequence=['green'],
        title='Top 5 Locations with Enquiries'
    )

    # Customize layout
     fig9.update_layout(
        xaxis_title='Location',
        yaxis_title='Number of Enquiries',
        plot_bgcolor='white',
        hovermode='x',
        width=1000,
        height=500
    )

    # Display in Streamlit
     st.plotly_chart(fig9, use_container_width=True)
     st.markdown(
     """
    ### Insights
    * <p style="color:blue"> Kozhikkode has highest not enrolled count with total number of 147 </p>
    * <p style="color:blue"> Have to improve marketing to other than malabar region of kerala,or if focusing only malabar its good performance of marketing
    """,
    unsafe_allow_html=True
     )
    elif cat=='Find_Us Base':
     st.subheader('Best Marketing Platform')
     find_us_counts = data['Find_us'].value_counts().reset_index()
     find_us_counts.columns = ['Find_us', 'Count']

# Create Plotly bar chart
     fig10 = px.bar(find_us_counts, 
               x='Find_us', 
               y='Count', 
               text='Count',
               title='Find Us Counts',
               color='Count',  # optional: color by count
               color_continuous_scale='Viridis')  # you can choose any color scale

     fig10.update_layout(
     xaxis_title='Find Us',
     yaxis_title='Count',
     template='plotly_white'
   )

# Show in Streamlit
     st.plotly_chart(fig10, use_container_width=True)
     st.markdown(
     """
     ### Insights
     * <p style="color:blue"> Most of joined enquiries are  through refferels, </p>
     * <p style="color:blue">Social media is also good platform for marketing</p>
     """,
     unsafe_allow_html=True
     )
    elif cat=='Qualification Base':
     st.subheader('Enquiry By Qualification')
     qual_counts = data['Qualification'].value_counts().reset_index()
     qual_counts.columns = ['Qualification', 'Count']

# Create Plotly pie chart
     fig11 = px.pie(qual_counts, 
               names='Qualification', 
               values='Count', 
               title='Number of enquiries by Qualification',
               hole=0)  # hole=0 for regular pie, >0 for donut

# Show in Streamlit
     st.plotly_chart(fig11, use_container_width=True)
     st.markdown(
    """
    * <b> <p style="color:orange"> 49% of enquiries have bachelor degree
    """,
    unsafe_allow_html=True
     )
    elif cat=='Issue Faced':
     st.subheader('Career issue by enrolls')
    # Prepare data: count of Career_issue where Status == 'joined'
     career_counts = data[data["Status"]=='joined'].groupby('Career_issue').size().reset_index(name='Count')

# Create Plotly bar chart
     fig12 = px.bar(career_counts, 
               x='Career_issue', 
               y='Count', 
               text='Count',
               title='Number of Joined Enquiries by Career Issue',
               color='Count',  # optional: color by count
               color_continuous_scale=['lightcoral', 'lightcoral'])  # keeps the same color

     fig12.update_layout(
     xaxis_title='Career Issue',
     yaxis_title='Count',
     template='plotly_white'
     )

# Show in Streamlit
     st.plotly_chart(fig12, use_container_width=True)
     st.markdown(
    """
    > <b> <p style="color:lightcoral"> Most of enrolls facing career gap and non it background(by available data of career issue)
    
    """,
    unsafe_allow_html=True
     )
    elif cat=='Looking_for based':
     st.subheader('Service Expected By joined Students')
     looking_counts = data[data['Status']=='joined'].groupby('Looking_for').size().reset_index(name='Count')

# Create Plotly bar chart
     fig13 = px.bar(looking_counts, 
               x='Looking_for', 
               y='Count', 
               text='Count',
               title='Number of Joined Enquiries by Looking For',
               color='Count',
               color_continuous_scale=['green', 'green'])  # keeps green color

     fig13.update_layout(
     xaxis_title='Looking For',
     yaxis_title='Count',
     template='plotly_white'
     )

# Show in Streamlit
     st.plotly_chart(fig13, use_container_width=True)
     st.markdown(
    """
    * <B> Most of enrolls excepting internship
    
    """,
    unsafe_allow_html=True
     )
    elif cat=='Technology Based':
     st.subheader('Most Enrolled Technology')
     tech_counts = data[data["Status"]=='joined'].groupby('Technology').size().reset_index(name='Count')

# Create Plotly bar chart
     fig14 = px.bar(tech_counts,
               x='Technology',
               y='Count',
               text='Count',
               title='Number of Joined Enquiries by Technology',
               color='Count',
               color_continuous_scale='Blues')  # optional: color scale

     fig14.update_layout(
     xaxis_title='Technology',
     yaxis_title='Count',
     template='plotly_white'
     )

# Show in Streamlit
     st.plotly_chart(fig14, use_container_width=True)
     st.markdown(
    """
    > <b> Most Enquiried and joined technology is python
    
    """,
    unsafe_allow_html=True
     )
     st.subheader('Technology Trend By year')
     df = data.groupby(['Year_Of_Enq', 'Technology']).size().unstack(fill_value=0)

# Create a figure
     fig15 = go.Figure()

# Add a line for each technology
     for tech in df.columns:
        fig15.add_trace(go.Scatter(
        x=df.index,
        y=df[tech],
        mode='lines+markers',
        name=tech
     ))

# Update layout
     fig15.update_layout(
     title='Technology Trends Over Time',
     xaxis_title='Year',
     yaxis_title='Count',
     template='plotly_white',
     hovermode='x unified',
     )

# Show in Streamlit
     st.plotly_chart(fig15, use_container_width=True)
     st.markdown(
    """
    ## Insights
    * Python dominates overall — peaking in 2022 (90) and staying strong through 2025.

    * Data Science surged from 0 in 2022 to 15 in 2025, showing fast-growing interest.

    * MERN Stack steadily rose, hitting its highest count (18) in 2025.
 
    * Flutter saw a sharp rise in 2022 (22) but slightly declined afterward.

    * Angular peaked in 2022 (6) and then dropped to zero — indicating a loss of popularity.

    * PHP declined after 2021, suggesting reduced new enrollments.
  
    * Web Designing fell to almost zero by 2024–2025, signaling waning demand.

    * AI and DevOps appeared only from 2024 onward — both emerging technologies.

    * 2024 was the overall peak year for most technologies before a mild drop in 2025.

    * Overall trend: shift from traditional stacks (PHP, Angular, Web Design) to modern, data-driven and full-stack fields (Python, Data Science, MERN, DevOps).

    
    """,
    unsafe_allow_html=True
    )
    elif cat=='Mod of study':
     st.subheader('mod of study enrolls')
     dfenroll=data[data['Status']=='yes'].groupby(data['Mode']).size()
     dfenroll=pd.DataFrame(dfenroll)
     dfenroll=dfenroll.rename(columns={0:'Count'})
     st.dataframe(dfenroll)
     st.markdown(
    """
    > <b>227 Enrolls re preferd Offline Class,Most of enrolls excpecting offline class

    """,
    unsafe_allow_html=True
     )
    elif cat=='Time Slot':
     st.subheader('Perferd Time slot by enrolls')
     dftime=data[data['Status']=='yes'].groupby(data['Time_Slot']).size()
     dftime=pd.DataFrame(dftime)
     dftime=dftime.rename(columns={0:'Count'})
     st.dataframe(dftime)
     st.markdown(
     """
    > <b> Most Of em prefering morning section

    """,
    unsafe_allow_html=True
     )
    elif cat=='Attended Staff':
     st.subheader('Attended Staffs and enrolls')
     total_counts = data['Attended_by'].value_counts().reset_index()
     total_counts.columns = ['Attended_by', 'Total_Enquiry']

     joined_counts = data[data['Status']=='joined'].groupby('Attended_by').size().reset_index(name='Joined_Enq')

# Merge total and joined counts
     merged_counts = pd.merge(total_counts, joined_counts, on='Attended_by', how='left')
     merged_counts['Joined_Enq'] = merged_counts['Joined_Enq'].fillna(0)

# Convert to long format for Plotly
     long_df = merged_counts.melt(id_vars='Attended_by', value_vars=['Total_Enquiry', 'Joined_Enq'],
                             var_name='Type', value_name='Count')

# Create Plotly grouped bar chart
     fig16 = px.bar(long_df,
               x='Attended_by',
               y='Count',
               color='Type',
               barmode='group',
               text='Count',
               title='Total vs Joined Enquiries by Attended By')

     fig16.update_layout(
      xaxis_title='Attended By',
      yaxis_title='Count',
      template='plotly_white'
     )

# Show in Streamlit
     st.plotly_chart(fig16, use_container_width=True)
     st.markdown(
    """
   * Highest Contribution: Baiju handled a significantly higher number of enquiries (289), accounting for the majority of total activity and indicating a   key role in attendance or enquiry handling.

   * Moderate Participation: Anupama (84) and Sivaprasad (44) contributed moderately, suggesting consistent but secondary levels of engagement compared to  Baiju.

   * Low Participation Group: The rest of the team — including Rifna (9), Rifana (7), Monish (4), Shafeeque (3), Faisal (3), Lekha (3), Deepak (2),     Praveena (1), Athulya (1), and Remin (1) — had very low enquiry counts, showing limited involvement.

    * Minimal Involvement: Shiju attended only once, indicating that while previously recorded as the only 100% attender, their total contribution to enquiries is minimal.
    * Overall Pattern: The data shows a highly skewed distribution, with one dominant contributor (Baiju) and several participants with minimal or one-time involvement — suggesting potential workload imbalance or irregular attendance across the team.
    
    """,
    unsafe_allow_html=True
     )
    elif cat=='Revenue Details':
     value = data.groupby('Status')['Fee_Details'].sum().iloc[1]
     st.metric(label="Total Fee Expected ", value=f"{value:,.2f}")
     st.subheader('Year Base Revenue')
     dfr=data.groupby(['Status','Year_Of_Enq'])['Fee_Details'].sum()
     dfr=pd.DataFrame(dfr)
     dfr=dfr.drop(['no'],axis=0,)
     st.dataframe(dfr)
    elif cat=='duration':
     st.subheader('Course duration and Enrollment')
     duration_counts = data[data['Status']=='joined'].groupby('Course_duration').size().reset_index(name='Count')

# Create Plotly bar chart
     fig17 = px.bar(duration_counts,
               x='Course_duration',
               y='Count',
               text='Count',
               title='Number of Joined Enquiries by Duration (months)',
               color='Count',
               color_continuous_scale='Teal')  # optional: color scale

# Update layout
     fig17.update_layout(
     xaxis_title='Duration (months)',
     yaxis_title='Count',
     template='plotly_white'
     )
# Show in Streamlit
     st.plotly_chart(fig17, use_container_width=True)

elif menu=="🤖 Prediction Model":
  st.header("ML Model For Predicting weather Enquiry will enroll or not")
  qualification_map = {'b.tech': 0, 'diploma': 1, 'other': 2, 'pg': 3,'ug':4}
  technology_map = {'angular':0,'artificial intelligence':1,'business analytics':2,'data analysis':3,'data science':4,
                    'devops':5,'flutter':6,'java':7,'javascript':8,'mern':9,'other':10,'php':11,'python':12,'react':13,'web designing':14}
  mode_map = {'offline': 1, 'online': 2, 'hybrid': 0}
  time_slot_map = {'afternoon': 0, 'evening': 1, 'morning': 2,'weekend':3}
  looking_for_map = {'courses': 0, 'internship': 1, 'other': 2, 'work experience': 3}
  find_us_map = {'google': 0, 'other': 1, 'reference': 2, 'social media': 3}
  career_issue_map = {'backpapers': 0, 'career-gap': 1, 'non-it': 2}
  attended_by_map={'anupama':0,'athulya':1,'baiju':2,'deepak':3,'faisal':4,'lekha':5,'monish':6,'praveena':7,'remin':8,'rifana':9,'rifna':10,
                   'shafeeque':11,'shiju':12,'sivaprasad':13}
  remark_map={'cancelled':0,'faculty shortage':1,'high fee':2,'higher study':3,'joined':4,'lost':5,'low fee':6,
              'not interested':7,'not intrested':8,'np':9,'other':10,'sample enquiry':11,'schedule issue':12,'suspected-rivals':13}
  with st.form('prediction form'):
   st.write("## Student Enqiury Details")
   name = st.text_input("Student Name", placeholder="Enter student name") 
   location = st.text_input("Location", placeholder="Enter location") 
   mobile_no = st.text_input("Mobile Number", placeholder="Enter mobile number") 
   email = st.text_input("Email", placeholder="Enter email address")

   col1,col2=st.columns(2)
   with col1:
     find_us = st.selectbox("How did you find us?", options=sorted(data['Find_us'].dropna().unique()))
     qualification=st.selectbox("Qulification?", options=sorted(data['Qualification'].dropna().unique()))
     career_issue=st.selectbox("career issue?", options=sorted(data['Career_issue'].dropna().unique()))
     lookingfor=st.selectbox("looking for", options=sorted(data['Looking_for'].dropna().unique()))
     technology=st.selectbox("technology", options=sorted(data['Technology'].dropna().unique()))
     mode=st.selectbox('mode',options=sorted(data['Mode'].dropna().unique()))
   with col2:
     timeslot=st.selectbox('timeslot',options=sorted(data['Time_Slot'].dropna().unique()))
     attendedby=st.selectbox('attendedby',options=sorted(data['Attended_by'].dropna().unique()))
     nopersons = st.number_input("Number of Persons", min_value=1, max_value=100, value=1)
     yop=st.number_input("year off passot", min_value=1990, max_value=2030, value=2025)
     yee=st.number_input("year off enquiry", min_value=1990, max_value=2030, value=2025)
     duration=st.number_input("duration of course(month)", min_value=1, max_value=12, value=6)
   
   
   submitted=st.form_submit_button('Predict')
   if submitted:
      input_data = {
        'Find_us': find_us_map.get(find_us, -1),
        'Qualification': qualification_map.get(qualification, -1),
        'Career_issue': career_issue_map.get(career_issue, -1),
        'Looking_for': looking_for_map.get(lookingfor, -1),
        'Technology': technology_map.get(technology, -1),
        'Mode': mode_map.get(mode, -1),
        'Time_Slot': time_slot_map.get(timeslot, -1),
        'Attended_by':attended_by_map.get(attendedby),
        'No_persons': nopersons,
        'Year_Of_pass_out': yop,
        'Year_Of_Enq': yee,
        'Course_duration': duration,
      }  
      input_df = pd.DataFrame([input_data])
      prediction = model.predict(input_df)[0]
      
      if prediction == 1: 
        st.success("✅ *WILL JOIN*") 
      else: 
        st.error("❌ *WILL NOT JOIN*")
      
    



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from datetime import datetime
import sqlite3
from scipy.stats import linregress
import seaborn as sns
os.chdir('/Users/josephlatona/Desktop/Homeruns_Project')
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 180)

#First wanted to look at homeruns on an individual basis so I downloaded data that contains information on every homerun hit from the 2016 season to the 2019 season 

Homers_2019 = pd.read_csv('/Users/josephlatona/Desktop/Homeruns_Project/Homeruns_2019.csv')
Homers_2018 = pd.read_csv('/Users/josephlatona/Desktop/Homeruns_Project/Homeruns_2018.csv')
Homers_2017 = pd.read_csv('/Users/josephlatona/Desktop/Homeruns_Project/Homeruns_2017.csv')
Homers_2016 = pd.read_csv('/Users/josephlatona/Desktop/Homeruns_Project/Homeruns_2016.csv')



#All the homerun data was merged into one table
All_homers=pd.concat([Homers_2016,Homers_2017,Homers_2018,Homers_2019])  


All_homers['Date'] = pd.to_datetime(All_homers['Date'], format = "%m/%d/%y")
All_homers['Year']=All_homers['Date'].dt.year
All_homers['Month']=All_homers['Date'].dt.month
Homers_ind=All_homers.set_index(["Year"])
#print(All_homers.Batter.value_counts(ascending=True))
#print(All_homers.groupby(['Year','Month'])['Tm'].agg(lambda x:x.value_counts().index[0]))
#print(All_homers.groupby(['Year','Month'])[['Batter','Tm']].agg(lambda x:x.value_counts().index[0]))






Hrs_by_yr_mnth = All_homers['Date'].groupby([All_homers.Date.dt.year.rename('Year'),All_homers.Date.dt.month.rename('Month')]).agg('count')

Hrs_by_yr_mnth.plot(kind='bar')
plt.xlabel("Year, Month")
plt.ylabel("Homeruns Hit")
plt.title("Distribution of Homeruns Hit w/ Year,Month Since 2016")


ax=Hrs_by_yr_mnth.plot(kind='bar')

def add_value_labels(ax, spacing=10):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = (y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# Call the function above. All the magic happens there.
add_value_labels(ax)
#plt.show()

#Now I want to look at some of the Statcast advanced data on a yearly basis for each team
file='Statcast_Data.xlsx'
statcast=pd.ExcelFile(file)
#check to make sure sheetnames were properly imported
#print(statcast.sheet_names)
#Now make each statcast data sheet a dataframe

statcast_2016=statcast.parse('2016')
statcast_2017=statcast.parse('2017')
statcast_2018=statcast.parse('2018')
statcast_2019=statcast.parse('2019')

#Now I want to look at the more advanced metrics to see if there were any shifts in trend over the past seasons. I will start by looking at launch angle 
angle_2016=statcast_2016[['Team','Launch Angle']]
angle_2017=statcast_2017[['Team','Launch Angle']]
angle_2018=statcast_2018[['Team','Launch Angle']]
angle_2019=statcast_2019[['Team','Launch Angle']]

#print(angle_2016['Launch Angle'].agg(np.mean))
#print(angle_2017['Launch Angle'].agg(np.mean))
#print(angle_2018['Launch Angle'].agg(np.mean))
#print(angle_2019['Launch Angle'].agg(np.mean))


def advstat(year,stat):
	df=year[['Team',stat]]
	return df


#print(advstat(statcast_2016,'Launch Angle').agg(np.mean))
#print(advstat(statcast_2016,'Barrel %').agg(np.mean))
#print(advstat(statcast_2017,'Launch Angle').agg(np.mean))
#print(advstat(statcast_2017,'Barrel %').agg(np.mean))
#print(advstat(statcast_2018,'Launch Angle').agg(np.mean))
#print(advstat(statcast_2018,'Barrel %').agg(np.mean))
#print(advstat(statcast_2019,'Launch Angle').agg(np.mean))
#print(advstat(statcast_2019,'Barrel %').agg(np.mean))

print(statcast_2016['SLG'].agg(np.mean))
print(statcast_2017['SLG'].agg(np.mean))
print(statcast_2018['SLG'].agg(np.mean))
print(statcast_2019['SLG'].agg(np.mean))

Angs = [['2016',statcast_2016['Launch Angle'].agg(np.mean)],['2017',statcast_2017['Launch Angle'].agg(np.mean)],['2018',statcast_2018['Launch Angle'].agg(np.mean)],['2019',statcast_2019['Launch Angle'].agg(np.mean)]]
Barrels = [['2016',statcast_2016['Barrel %'].agg(np.mean)],['2017',statcast_2017['Barrel %'].agg(np.mean)],['2018',statcast_2018['Barrel %'].agg(np.mean)],['2019',statcast_2019['Barrel %'].agg(np.mean)]]
Launch_Angles_yr=pd.DataFrame(Angs,columns=['Year','Avg Launch Angle'])
Barrels_yr = pd.DataFrame(Barrels,columns=['Year','Avg Barrels %'])

print(Launch_Angles_yr)

#plt.figure()
#All_homers.groupby('Year')['Year'].agg('count').plot()

#fig,axs=plt.subplots()
#axs.plot(Launch_Angles_yr['Year'],Launch_Angles_yr['Avg Launch Angle'],color='r',marker='o')
#ax2=axs.twinx()
# make a plot with different y-axis using second axis object
#All_homers.groupby('Year')['Year'].agg('count').plot(kind='bar',alpha=.5)
#plt.title('Avg. Launch Angle & Total Hrs by Year')

#fig,axs=plt.subplots()
#axs.plot(Barrels_yr['Year'],Barrels_yr['Avg Barrels %'],color='r',marker='o')
#ax2=axs.twinx()
# make a plot with different y-axis using second axis object
#All_homers.groupby('Year')['Year'].agg('count').plot(kind='bar',alpha=.5)
#plt.title('Avg. Barrel % & Total Hrs by Year')




fig2,ax3=plt.subplots(2,2)
ax3[0,0].plot(statcast_2016['Team'],statcast_2016['Launch Angle'],color='r')
ax3[0,0].set_ylabel('Launch Angle %')

ax4=ax3[0,0].twinx()
plt.bar(statcast_2016['Team'],statcast_2016['HR'],alpha=0.5)
plt.xlabel('Teams')
plt.title('2016 Homeruns vs. Launch Angle')


ax3[1,0].plot(statcast_2017['Team'],statcast_2017['Launch Angle'],color='r')
ax3[1,0].set_ylabel('Launch Angle %')
ax5=ax3[1,0].twinx()
plt.bar(statcast_2017['Team'],statcast_2017['HR'],alpha=0.5)
plt.title('2017 Homeruns vs. Launch Angle')

ax3[0,1].plot(statcast_2018['Team'],statcast_2018['Launch Angle'],color='r')
ax6=ax3[0,1].twinx()
plt.bar(statcast_2018['Team'],statcast_2018['HR'],alpha=0.5)
plt.xlabel('Teams')
plt.ylabel('Total Homeruns Hit')
plt.title('2018 Homeruns vs. Launch Angle')

ax3[1,1].plot(statcast_2019['Team'],statcast_2019['Launch Angle'],color='r')
ax7=ax3[1,1].twinx()
plt.bar(statcast_2019['Team'],statcast_2019['HR'],alpha=0.5)
plt.ylabel('Total Homeruns Hit')
plt.title('2019 Homeruns vs. Launch Angle')

plt.setp(ax3[0,0].get_xticklabels() + ax3[0,1].get_xticklabels() + ax3[1,0].get_xticklabels() + ax3[1,1].get_xticklabels()  , rotation=90)
ax3[0,0].tick_params(axis='x', which='major', width= 2,labelsize=8)
ax3[0,1].tick_params(axis='x', which='major', width= 2,labelsize=8) 
ax3[1,0].tick_params(axis='x', which='major', width= 2,labelsize=8)
ax3[1,1].tick_params(axis='x', which='major', width= 2,labelsize=8)
plt.tight_layout()
plt.show()


plt.figure()
print(statcast_2019[['HR','Launch Angle']].corr())
res=linregress(statcast_2019['HR'],statcast_2019['Launch Angle'])
print(linregress(statcast_2019['HR'],statcast_2019['Launch Angle']))
fx= np.array([statcast_2019['HR'].min(),statcast_2019['HR'].max()])
fy=res.intercept+ res.slope* fx
plt.plot(statcast_2019['HR'],statcast_2019['Launch Angle'],'o')
plt.plot(fx,fy,'-',alpha=0.7)
plt.title('Launch Angle vs Homeruns 2019')
plt.figure()
plt.plot(statcast_2019['HR'],statcast_2019['Barrel %'],'o')
res1=linregress(statcast_2019['HR'],statcast_2019['Barrel %'])
fy1=res1.intercept+ res1.slope* fx
plt.plot(fx,fy1,'-')

plt.figure()
sns.countplot(x='Tm',data=Homers_2019)



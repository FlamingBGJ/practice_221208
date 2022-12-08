# 라이브러리에서 제공하는 데이터셋을 활용해 그래프를 그리는 연습을 할 것이다.

# 라이브러리 호출
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 타이타닉 데이터셋 가져오기
titanic = sns.load_dataset("titanic")

# 타이타닉 데이터셋 살펴보기
print(titanic.head())
print(titanic.info())


# 1. 회귀선이 있는 산점도
sns. set_style("darkgrid")
fig = plt.figure(figsize=(15,5))
ax_1 = fig.add_subplot(1,2,1)
ax_2 = fig.add_subplot(1,2,2)
#그래프 그리기 - 선형회귀선 표시(fit_reg=True)
sns.regplot(x = "age",
            y = "fare",
            data=titanic,
            ax = ax_1)
# 그래프 그리기 - 선형회귀선 미표시(fig_reg=false)
sns.regplot(x = "age",
            y = "fare",
            data=titanic,
            ax = ax_2,
            fit_reg=False,
            color ="r")
plt.show()


# 2. 히스토그램/커널 밀도 그래프
fig  = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
sns.displot(titanic["fare"], ax = ax1) # displot
sns.kdeplot(x="fare", data = titanic, ax=ax2) # kedplot
sns.histplot(x = "fare", data = titanic, ax=ax3) # histplot
plt.show()


# 3. 히트맵
table = titanic.pivot_table(index = ["sex"], columns = ["class"], aggfunc = ["size"])
sns.heatmap(table,
            annot=True, fmt = "d",
            cmap = "YlGnBu",
            linewidth = "5",
            cbar = False)
plt.show()

# 4. 범주형 데이터의 산점도
sns.set_style("whitegrid")
fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
# 이산형 변수의 분포 - 데이터 분산 미고려(중복표시 O)
sns.stripplot(x = "class",
                y = "age", 
                data = titanic,
                ax = ax1, color = "orange")
# 이산형 변수의 분포 - 데이터 분산 고려(중복표시 x)
sns.swarmplot(x = "class", 
                y = "age", 
                data = titanic,
                ax = ax2, color = "green")
# 차트 제목 표시
ax1.set_title("Strip Plot")
ax2.set_title("Strip plot")
plt.show()

# 5. 막대그래프
fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
# x축, y축에 변수 할당
sns.barplot(x = "sex", y = "survived", data = titanic,ax = ax1)
# x축, y축에 변수 할당하고 hue 옵션 추가
sns.barplot(x = "sex", y = "survived", hue = "class", data = titanic,ax = ax2)
# x축, y축에 변수 할당하고 hue 옵션 추가하여 누적 출력
sns.barplot(x = "sex", y = "survived", hue = "class", dodge="false", data = titanic,ax = ax3)
# 차트 제목 표시
ax1.set_title("titanic class")
ax2.set_title("titanic class - who")
ax2.set_title("titanic class - who(stacked)")
plt.show()


# 6. 박스 플롯/바이올린 그래프
fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
# 박스플롯 기본값
sns.boxplot(x="alive", y="age", data=titanic, ax=ax1)
# 박스플롯 - hue 변수 추가
sns.boxplot(x="alive", y="age",hue="sex", data=titanic, ax=ax2)
# 바이올린플롯 기본값
sns.violinplot(x="alive", y="age", data=titanic, ax=ax3)
# 바이올린플롯 - hue 변수 추가
sns.violinplot(x="alive", y="age",hue="sex", data=titanic, ax=ax4)
plt.show()


# 7. 조인트 그래프
# 조인트 그래프 - 산점도(기본값)
j1 = sns.jointplot(x="fare",y="age",data=titanic)
# 조인트 그래프 - 회귀선
j2 = sns.jointplot(x="fare",y="age",kind="reg",data=titanic)
# 조인트 그래프 - 육각 그래프
j3 = sns.jointplot(x="fare",y="age",kind="hex",data=titanic)
# 조인트 그래프 - 커널 밀집 그래프
j4 = sns.jointplot(x="fare",y="age",kind="kde",data=titanic)
# 차트 제목 표시
ax1.set_title("titanic fare - scatter",size = 15)
ax2.set_title("titanic fare - reg",size = 15)
ax3.set_title("titanic fare - hex",size = 15)
ax4.set_title("titanic fare - kde",size = 15)
plt.show()


# 8. 화면 그리드 분할
# 조건에 따라 그리드 나누기
g = sns.FacetGrid(data = titanic,col="who",row="survived")
# 그래프 적용하기
g = g.map(plt.hist,"age")


# 9. 이변수 데이터의 분포
# 이변수 데이터 분포
# 타이타닉 데이터 셋 중에서 분석
titanic_pair = titanic[["age","pclass","fare"]]
# 조건에 따라 그리드 나누기
g = sns.pairplot(titanic_pair)
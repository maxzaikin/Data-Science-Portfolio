# Data-Science-Portfolio
Hello and welcome to my Data Science portfolio! This repository serves as a central hub showcasing a collection of my projects demonstrating skills and experience in data analysis, machine learning, and artificial intelligence. Each project folder contains Jupyter Notebooks with detailed code, explanations, and visualizations.

| Название проекта | Сфера деятельности | Направление деятельности | Навыки и инструменты | Метрики качества |Методы визуального анализа | Задачи проекта | Описание проекта | Ключевые слова проекта |
|---|---|---|---|---|---|---|---|---|
| [1. Исследование данных сервиса “Яндекс.Музыка” — сравнение пользователей двух городов](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Data-Analysis/1.%20Data_Analysis_of_Yandex.Music_Comparing_Users_in_Two_Cities.ipynb)| Интернет-сервисы; Стриминговый сервис |Data Analyst |Python; Pandas |Сотрировки, группировки, анализ уникальности| Текстовый вывод |На реальных данных Яндекс.Музыки c помощью библиотеки Pandas и её возможностей проверить данные и сравнить поведение и предпочтения пользователей двух столиц — Москвы и Санкт-Петербурга. | На реальных данных Яндекс.Музыки вы проверите данные и сравните поведение пользователей двух столиц.|data analyst; аналитик данных; аналитик; analyst |
|[2. Исследование надёжности заёмщиков — анализ банковских данных](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Data-Analysis/2.%20Bank_Loan_Default_Risk_Study.ipynb) |Банковская сфера;Кредитование|Data Analyst; Финансовый аналитик | EDA; Python; Pandas|Сотрировки, группировки, анализ уникальности| Текстовый анализ|На основе статистики о платёжеспособности клиентов исследовать влияет ли семейное положение и количество детей клиента на факт возврата кредита в срок|На основе данных кредитного отдела банка исследовал влияние семейного положения и количества детей на факт погашения кредита в срок. Была получена информация о данных. Определены и обработаны пропуски. Заменены типы данных на соответствующие хранящимся данным. Удалены дубликаты. Категоризованы данные. Один датафрейм декомпозирован на три. | data analyst, налитик данных, аналитик, финансовый аналитик, analyst|
|[3. Продажа квартир в Санкт-Петербурге — анализ рынка недвижимости](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Data-Analysis/3.%20Research_on_the_Real_Estate_Market_for_Apartment_Sales.ipynb) | Интернет-сервисы; Площадки объявлений| Маркетинг-аналитик; Fraud-аналитик; Data Analyst| Python; Pandas; Matplotlib; EDA; визуализация данных; предобработка данных; Конвертация типов серий;Обработка пропусков; Анализ на уникальность;Визуализация плотности точек данных на плоскости|tbd|Столбчатая гистограмма; Круговая гистрограмма; Гексагональная бининг-диаграмма; Наложение гистограм; Ящик с усами;Диаграмма рассеяния; Матрица диаграм рассеяния(matplot)| Используя данные сервиса Яндекс.Недвижимость, определить рыночную стоимость объектов недвижимости и типичные параметры квартир | На основе данных сервиса Яндекс.Недвижимость определена рыночная стоимость объектов недвижимости разного типа, типичные параметры квартир, в зависимости от удаленности от центра. Проведена редобработка данных. Добавлены новые данные. Построены гистограммы, боксплоты, диаграммы рассеивания.| маркетинговый аналитик; фрод аналитик; fraud analyst; data analyst; аналитик данных;  аналитик; analyst; обработка данных; histogram; boxplot; scattermatrix; категоризация; scatterplot; фрод-мониторинг|
|[4. Определение выгодного тарифа для телеком компании](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Machine-Learning/4.%20Determining_a_Profitable_Rate_for_a_Telecom_Company.ipynb)|Телеком|Маркетинг-аналитик; Продуктовый аналитик; Data Analyst|Python; Pandas; Matplotlib; Seaborn; NumPy; SciPy; описательная статистика; проверка статистических гипотез; Дерево решений; Случайный лес; Логистическая регрессия; Подбор гиперпараметра|accuracy;precission;recall|Матрица диаграм рассеяния(seaborn); Сравнительный анализ моделей|  На основе данных клиентов оператора сотовой связи проанализировать поведение клиентов и поиск оптимального тарифа|Проведен предварительный анализ использования тарифов на выборке клиентов, проанализировано поведение клиентов при использовании услуг оператора и рекомендованы оптимальные наборы услуг для пользователей. Проведена предобработка данных, их анализ. Проверены гипотезы о различии выручки абонентов разных тарифов и различии выручки абонентов из Москвы и других регионов.|аналитик; analyst; аналитик данных; data analyst|
| [8. Определение наиболее выгодного региона нефтедобычи](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Machine-Learning/8.%20Determining_the_Most_Profitable_Oil_Production_Region.ipynb)|Добывающие компании | Машинное обучение; Регррессия; Разработка; бизнес-модели; Финансовый аналитик; Расчет прибыли и рисков|Matplotlib; Seaborn; Pandas; Scikit-learn; бутстреп;Линейная зависимость;Нелинейная зависимость;полиномиальная регрессия  |RMSE; R2|Boxplot; Histogram; Матрица диаграм рассеяния; Матрица корелияции; Масштабирование признаков; Анализ остатков модели(Диаграма рассеяния, Гистограмма распределения); Наложение диаграм|  На основе данных геологи разведки выбрать район добычи нефти|По предоставленым данным пробы нефти в трёх регионах. Построить модель для определения региона, где добыча принесёт наибольшую прибыль. | аналитик; analyst; аналитик данных; data analyst; data scientist; ML Engineer|
| [15. Прогнозирование количества заказов такси на следующий час.](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Machine-Learning/15.%20Forecasting_Taxi_Orders_for_the_Next_Hour.ipynb)|Бизнес; Интернет-сервисы;Стартапы | Машинное обучение|Python; Pandas; Scikit-learn; statsmodels; RandomizedSearchCV; LinearRegression; SARIMA; SARIMAX Time series; Скользящее среднее; Анализ на монотонность; Аггрегация временных рядов;Feature Engineering; Pipeline|ADF(Augmented Dicke-Fuller); KPSS; RMSE | график частичной автокорреляции(PACF); ACF; Линейный график; Scatter plot; Динамика роста; Тренды; Сезонность; Шумы; Анализ на стационарность| Компания такси собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Строится модель для такого предсказания. |Задача: построить модель для прогноза количества заказов такси на следующий час. |временные ряды; регрессия; предсказания |
|[20. Оптимизация процесса восстановление золота из руды](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Machine-Learning/20.%20Gold_Recovery_From_Ore.ipynb) |Горнодобывающая промышленность |Машинное обучение; Регррессия; Разработка; бизнес-модели; Финансовый аналитик; Расчет коэфициента обогощения руды |Pandas; Scikit-learn; Matplotlib; NumPy; SciPy; Линейная Регрессиия; Решающее дерево(DecisionTreeRegressor); Случайный лес(RandomForestRegressor); DummyRegressor|МАЕ(самостоятельная реализация);SMAPE|Гистограммы; BoxPlot; HeatMap(Матрица кореляции); График регрессии c доверительным интервалом (Seaborn.regplot); Гистограммы с плотностью ядра (KDE); графики оценки плотности ядра (KDE) Анализ распределения размеров гранул на обучающей и тестовой выборках;| На основе предоставленных данных разработать модель оптимизирующую эффективность процесса флотации драгоценных металлов|Разработка прототипа модели машинного обучения для заказчикаЖ Компания «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий. Модель предсказает коэффициент восстановления золота из золотосодержащей руды на основе предоставленных производственных данных с параметрами добычи и очистки. | аналитик; analyst; аналитик данных; data analyst; data scientist; ML Engineer|
|[21. Прогнозирование удоя и вкуса молока](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Machine-Learning/21.%20Predict_Milk_Taste_Characteristics_and_Milk_Yield.ipynb) | Сельское хозяйство| Машинное обучение; Логистическая регрессия; ; Классификация; Разработка; бизнес-модели; Предсказание заданных качеств |Pandas; Scikit-learn; Matplotlib; NumPy; SciPy;Pipeline; OneHotEncoder; Feature Engineering; Нелинейная зависимость;Метод Спирмена; Масштабирование признаков;|R2;MSE;MAE;RMSE;Accuracy; Precission; Recall|Barplot(сравнение метрик качества);Pie chart; Histogram; BoxPlot; Heatmap; Диаграммы рассеяния;Scatter plot; Анализ остатков; Матрица ошибок(Confusion matrix); график кривой Precision-Recall для модели логистической регрессии| Заказчик хочет, чтобы каждая корова давала не менее 6000 килограммов молока в год, а её надой был вкусным — строго не хуже заданных характеристик качества |Разработать модель МО, которая поможет управлять рисками и принимать объективное решение о покупке. Поставщик, коров «ЭкоФерма» предоставил подробные данные о своих стадах. Необходимо создать две прогнозные модели для отбора коров в поголовье заказчика: 1. Модель МО для прогноза возможного удоя коровы (целевой признак Удой); 2. Модель МО для рассчита вероятности получить молоко удовлетворяющее заданным органолептическим характеристикам заказчика (целевой признак Вкус молока). | аналитик; analyst; аналитик данных; data analyst; data scientist; ML Engineer|
|[22. Исследование снижения покупательской активности](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Machine-Learning/22.%20Research_on_Factors_Contributing_to_Reduced_Consumer_Spending.ipynb) | Интернет-сервисы; Онлайн магазины|Машинное обучение; Регррессия; Разработка; бизнес-модели  |Pandas; Scikit-learn; SimpleImputer; OneHotEncoder;OrdinalEncoder; StandardScaler;DecissionTreeClassifier; KNeighborsClassifier;SVC; LogisticRegression; RandomizedSearchCV; Matplotlib; NumPy; SciPy;Pipeline |ROC-AUC|График ROC-AUC для логистической регрессии; Barplot; Piechart; Boxplot; Гистограммы распределения с наложением(seaborn); Heatmp; Анализ важности признаков; Сегментация покупателей; Анализ распределения с наложением| Разработать решение, которое позволит персонализировать предложения постоянным клиентам, чтобы увеличить их покупательскую активность.| Интернет-магазин «В один клик» продаёт разные товары: для детей, для дома, мелкую бытовую технику, косметику и даже продукты. Отчёт магазина за прошлый период показал, что активность покупателей начала снижаться. | аналитик; analyst; аналитик данных; data analyst; data scientist; ML Engineer|
|[23. Исследование продаж компьютерных игр](https://github.com/maxzaikin/Data-Science-Portfolio/blob/main/Data-Analysis/23.%20Video_Game_Sales_Research.ipynb)| Маркетинг-аналитик; Продуктовый аналитик; Data Analyst| описательная статистика; проверка статистических гипотез |Python; Pandas; Matplotlib; NumPy; SciPy; Pivot tables |не применимо |Barplot; Линейные графики с наложением; Boxplot(seaborn); Scatter plot; Hexbin| Интернет-магазин заказал провести исследование по выявлению закономерностей определяющиех успешность игры. | На  данных продажам игр выявить потенциально популярные игры |data analyst; аналитик данных; аналитик; analyst |
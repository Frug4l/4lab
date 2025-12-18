import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # для статистических тестов

# 1. загрузка и предварительный просмотр данных
print("=" * 50)
print("ШАГ 1: ЗАГРУЗКА ДАННЫХ")
print("=" * 50)

df = pd.read_csv('WHR2023.csv')
print(f"Размер датасета: {df.shape}")
print(f"\nНазвания столбцов:\n{list(df.columns)}")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nБазовая информация:")
print(df.info())
print(f"\nПропущенные значения:\n{df.isnull().sum()}")

# очистка данных от строк с пропусками
df_clean = df.dropna().copy()
print(f"\nРазмер после очистки: {df_clean.shape}")

# переименуем ключевые столбцы для удобства
column_map = {
    'Country name': 'country',
    'Ladder score': 'happiness_score',
    'Logged GDP per capita': 'log_gdp',
    'Social support': 'social_support',
    'Healthy life expectancy': 'life_expectancy',
    'Freedom to make life choices': 'freedom',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}
df_clean.rename(columns=column_map, inplace=True)

# 2. основной статистический анализ
print("\n" + "=" * 50)
print("ШАГ 2: СТАТИСТИЧЕСКИЙ АНАЛИЗ")
print("=" * 50)

key_columns = ['happiness_score', 'log_gdp', 'social_support', 'life_expectancy', 'freedom', 'generosity', 'corruption']
print("Описательная статистика для ключевых показателей:")
print(df_clean[key_columns].describe().round(3))

# 3. проверка статистических гипотез
print("\n" + "=" * 50)
print("ШАГ 3: ПРОВЕРКА ГИПОТЕЗ")
print("=" * 50)

# гипотеза 1: распределение уровня счастья является нормальным
score = df_clean['happiness_score']
stat, p_value = stats.shapiro(score.sample(min(5000, len(score)), random_state=42))
print(f"\nГипотеза 1: Нормальность распределения 'happiness_score'")
print(f"Тест Шапиро-Уилка: статистика = {stat:.4f}, p-value = {p_value:.4f}")
if p_value > 0.05:
    print("Вывод: Распределение НЕ ОТЛИЧАЕТСЯ от нормального (p > 0.05).")
else:
    print("Вывод: Распределение ОТЛИЧАЕТСЯ от нормального (p <= 0.05).")

# гипотеза 2: существует значимая положительная корреляция между ввп и счастьем
corr, p_value_corr = stats.pearsonr(df_clean['log_gdp'], df_clean['happiness_score'])
print(f"\nГипотеза 2: Корреляция между log_gdp и happiness_score")
print(f"Корреляция Пирсона: r = {corr:.4f}, p-value = {p_value_corr:.4e}")
if p_value_corr < 0.05 and corr > 0:
    print("Вывод: Существует статистически значимая положительная корреляция.")
else:
    print("Вывод: Значимая положительная корреляция НЕ обнаружена.")

# 4. создание визуализаций (аналогично репозиторию)
print("\n" + "=" * 50)
print("ШАГ 4: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
print("=" * 50)

# настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 4.1. график распределения основного показателя (аналог price_plot)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Анализ данных World Happiness Report 2023', fontsize=16, fontweight='bold')

# график 1: распределение уровня счастья по странам (аналог price_plot)
axes[0, 0].plot(df_clean.sort_values('happiness_score')['happiness_score'].values, linewidth=2)
axes[0, 0].set_title('Упорядоченное распределение уровня счастья по странам', fontsize=12)
axes[0, 0].set_xlabel('Порядковый номер страны')
axes[0, 0].set_ylabel('Happiness Score')
axes[0, 0].grid(True, alpha=0.3)

# график 2: гистограмма распределения (аналог returns_hist)
axes[0, 1].hist(df_clean['happiness_score'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(df_clean['happiness_score'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Среднее: {df_clean["happiness_score"].mean():.2f}')
axes[0, 1].axvline(df_clean['happiness_score'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Медиана: {df_clean["happiness_score"].median():.2f}')
axes[0, 1].set_title('Гистограмма распределения уровня счастья', fontsize=12)
axes[0, 1].set_xlabel('Happiness Score')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# график 3: qq-plot для проверки нормальности (аналог qqplot)
stats.probplot(df_clean['happiness_score'], dist="norm", plot=axes[1, 0])
axes[1, 0].get_lines()[0].set_markerfacecolor('steelblue')
axes[1, 0].get_lines()[0].set_markeredgecolor('steelblue')
axes[1, 0].get_lines()[0].set_markersize(5.0)
axes[1, 0].get_lines()[1].set_color('crimson')
axes[1, 0].get_lines()[1].set_linewidth(2.0)
axes[1, 0].set_title('QQ-plot для проверки нормальности распределения', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# график 4: кумулятивное распределение / топ-n стран (аналог cum_return)
top_n = 15
top_countries = df_clean.nlargest(top_n, 'happiness_score').sort_values('happiness_score')
axes[1, 1].barh(range(top_n), top_countries['happiness_score'], color='orange', alpha=0.7)
axes[1, 1].set_yticks(range(top_n))
axes[1, 1].set_yticklabels(top_countries['country'])
axes[1, 1].set_title(f'Топ-{top_n} стран по уровню счастья', fontsize=12)
axes[1, 1].set_xlabel('Happiness Score')
axes[1, 1].invert_yaxis()  # чтобы страна №1 была наверху
for i, v in enumerate(top_countries['happiness_score']):
    axes[1, 1].text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('happiness_analysis_dashboard.png', dpi=150, bbox_inches='tight')
print("Датчик визуализаций сохранен как 'happiness_analysis_dashboard.png'")

# 5. дополнительная визуализация: матрица корреляций (важно для анализа)
print("\nСоздание дополнительной визуализации: тепловой карты корреляций...")
fig2, ax2 = plt.subplots(figsize=(10, 8))
corr_matrix = df_clean[key_columns].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax2)
ax2.set_title('Матрица корреляций ключевых показателей', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('happiness_correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("Тепловая карта корреляций сохранена как 'happiness_correlation_heatmap.png'")

# 6. сохранение результатов анализа в файл
print("\n" + "=" * 50)
print("ШАГ 5: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 50)

# сохраняем очищенные данные с новыми названиями столбцов
output_filename = 'world_happiness_cleaned_analyzed.csv'
df_clean.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Очищенный и обработанный датасет сохранен в файл: '{output_filename}'")

# создаем текстовый файл с основными выводами
with open('happiness_analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("ОСНОВНЫЕ ВЫВОДЫ АНАЛИЗА WORLD HAPPINESS REPORT 2023\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"1. ОБЩАЯ ИНФОРМАЦИЯ\n")
    f.write(f"   - Проанализировано стран: {len(df_clean)}\n")
    f.write(f"   - Средний уровень счастья: {df_clean['happiness_score'].mean():.3f}\n")
    f.write(f"   - Медианный уровень счастья: {df_clean['happiness_score'].median():.3f}\n")
    f.write(f"   - Стандартное отклонение: {df_clean['happiness_score'].std():.3f}\n\n")

    f.write(f"2. ПРОВЕРКА ГИПОТЕЗ\n")
    f.write(f"   - Нормальность распределения (Шапиро-Уилк): p-value = {p_value:.4f}\n")
    f.write(f"     Вывод: Распределение {'НЕ отличается' if p_value > 0.05 else 'отличается'} от нормального.\n")
    f.write(f"   - Корреляция ВВП и счастья (Пирсон): r = {corr:.4f}, p-value = {p_value_corr:.4e}\n")
    f.write(
        f"     Вывод: {'Обнаружена значимая положительная корреляция.' if p_value_corr < 0.05 and corr > 0 else 'Значимая корреляция не обнаружена.'}\n\n")

    f.write(f"3. ТОП-5 САМЫХ СЧАСТЛИВЫХ СТРАН\n")
    for i, row in df_clean.nlargest(5, 'happiness_score').iterrows():
        f.write(f"   {i + 1}. {row['country']}: {row['happiness_score']:.3f}\n")
    f.write(f"\n")

    f.write(f"4. САМЫЕ СИЛЬНЫЕ КОРРЕЛЯЦИИ С УРОВНЕМ СЧАСТЬЯ\n")
    happiness_corr = corr_matrix['happiness_score'].drop('happiness_score').sort_values(ascending=False)
    for factor, corr_value in happiness_corr.head(3).items():
        f.write(f"   - {factor}: {corr_value:.3f}\n")

print("Текстовый отчет с выводами сохранен как 'happiness_analysis_summary.txt'")

# 7. финальный просмотр результатов
print("\n" + "=" * 50)
print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
print("=" * 50)
print("\nСозданные файлы:")
print("  1. happiness_analysis_dashboard.png      - Датчик из 4 графиков")
print("  2. happiness_correlation_heatmap.png     - Тепловая карта корреляций")
print("  3. world_happiness_cleaned_analyzed.csv  - Очищенный датасет")
print("  4. happiness_analysis_summary.txt        - Текстовый отчет с выводами")

# показываем финальные графики
plt.show()

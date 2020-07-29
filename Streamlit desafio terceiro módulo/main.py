import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt


def main():
    st.title('Desafio terceira semana AceleraDev')
    st.subheader('Base de dados - Dota 2 - Partidas')
    st.image('dota_heroes.png', use_column_width=True)

    df = pd.read_csv("resultss.csv")

    if df is not None:
        df = df[df['account_id'] == 120236951]

        df = df[['kda', 'kills', 'deaths', 'assists', 'gold_per_min',
                 'tower_damage', 'xp_per_min', 'patch', 'last_hits', 'win','hero_damage','radiant_win']]
        df = df.reset_index(drop=True)

        st.write("Amostragem de partidas da conta de número 120236951")
        size = st.slider('Quantidade de valores na visualização da tabela', 0, df.shape[0], 5)
        st.dataframe(df.loc[0: size])

        UnivarStats(df)

        GraphView(df)
        


def GraphView(df):

    
    st.title("Visualização de dados")

    radio = st.radio("Selecione qual gráfico gostaria de visualizar",
                     ("Correlação", "Histograma", "Linha", "Desvio Padrão"))

    if radio == "Correlação":

        option = st.multiselect("Selecione o patch", df['patch'].unique())
        st.subheader("Matriz de correlação")

        st.altair_chart(CorrMatrix(df.loc[df['patch'].isin(option)]))

    if radio == "Histograma":

        option = st.selectbox("Selecione a coluna", df.columns)
        st.subheader("Histograma")

        st.altair_chart(Histogram(option, df))

    if radio == "Linha":

        option = st.selectbox("Selecione a coluna", df.columns)
        st.subheader("Linha")

        st.altair_chart(LineChart(option, df))

    if radio == "Desvio Padrão":
        st.write(df[option].std())

def UnivarStats(df):    
    
    st.title("Estatística Univariada")
    
    option = st.selectbox("Selecione uma coluna", df.columns)

    radio = st.radio("Selecione qual medida gostaria de analisar",
                     ("Média", "Mediana", "Moda", "Desvio Padrão"))

    if radio == "Média":
        st.write(df[option].mean())
    if radio == "Mediana":
        st.write(df[option].median())
    if radio == "Moda":
        st.write(df[option].mode()[0])
    if radio == "Desvio Padrão":
        st.write(df[option].std())


def CorrMatrix(df):
    df = df.drop(columns = 'win')

    df = (df.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))

    df['correlation_label'] = df['correlation'].map('{:.2f}'.format)
    base = alt.Chart(df, width = 600).encode(
        x='variable2:O',
        y='variable:O')

    text = base.mark_text().encode(
        text='correlation_label',
        color=alt.condition(
            alt.datum.correlation > 0.5,
            alt.value('white'),
            alt.value('black')
        ))

    cor_plot = base.mark_rect().encode(
        color='correlation:Q'
    )
    return cor_plot + text


def Histogram(coluna, df):

    chart = alt.Chart(df, width = 600).mark_bar().encode(
        alt.X(coluna, bin = True),
        alt.Y('count()', title = 'quantidade'),
        color = 'win:N',
    ).interactive()

    return chart;

def  LineChart(coluna,df):
    df_win = df[df['win']==1][coluna].dropna().value_counts().sort_index()
    df_matches = df[coluna].dropna().value_counts().sort_index()

    win_rate = (df_win // df_matches)

    base = alt.Chart(win_rate)

    line = base.mark_line().encode(
        x = win_rate.values,
        y = win_rate.index
    )

    return line
    

if __name__ == '__main__':
    main()

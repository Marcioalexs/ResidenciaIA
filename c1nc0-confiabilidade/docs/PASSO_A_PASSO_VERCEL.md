# C1NC0 — Naive Bayes V1 — publicação em produção

## 1. Pré-requisito indispensável: gerar os artefatos
Este pacote não contém o dataset original, portanto os binários treinados não podem ser fabricados com segurança a partir do notebook sozinho.

Na raiz `c1nc0-confiabilidade`, com o ambiente virtual ativado:

```powershell
python -m pip install -r requirements.txt
python -m ml.train_models "C:\CAMINHO\DATASET_TERCEIROGENITO_V1_URL_UNIFORME_PT.csv"
```

Confirme:

```powershell
Get-ChildItem .\models\
```

Devem existir `gaussian_nb_pipeline.joblib`, `tfidf_vectorizer.joblib`, `multinomial_nb.joblib` e `model_metadata.json`.

## 2. Copiar os arquivos deste ZIP
Copie/substitua na pasta `c1nc0-confiabilidade` do repositório, preservando a pasta `resources` existente.

## 3. Teste local
```powershell
python app.py
```
Abra `http://127.0.0.1:5000` e também `http://127.0.0.1:5000/health`.
`models_ready` deve ser `true`.

## 4. Commit na branch de integração
```powershell
git status
git add c1nc0-confiabilidade
git commit -m "feat: integrate Naive Bayes V1 into C1NC0 web prototype"
git push -u origin feature/integracao-naive-bayes-v1
```

## 5. Publicar diretamente em produção (decisão do grupo)
Depois do teste local, faça merge da branch na `main`:

```powershell
git checkout main
git pull origin main
git merge feature/integracao-naive-bayes-v1
git push origin main
```

Se o projeto Vercel estiver ligado à `main`, o push dispara o Production Deployment.

## 6. Configuração Vercel
No projeto, confira Settings > General:
- Production Branch: `main`
- Root Directory: `c1nc0-confiabilidade`

O pacote inclui `vercel.json`. Não são necessárias variáveis de ambiente para esta V1.

## 7. Validação em produção
1. Abra `/health` e confirme `models_ready: true`.
2. Teste URL válida com texto suficiente.
3. Teste URL com HTTP 403/404.
4. Teste URL inválida e IP privado.
5. Confira que a UI nunca apresenta a classe textual `true/fake` como veredito.
6. Confira que os percentuais são chamados de distribuição relativa de evidências.
7. Registre exemplos, limitações e feedback do piloto.

## Observação sobre LanguageTool
O notebook usa LanguageTool para `percentual_erros_ortograficos`. Em serverless, Java não é garantido. Nesta V1 Web a feature é marcada ausente e imputada pela mediana do pipeline GaussianNB. A interface mostra a feature como `N/A / imputada`. Isso deve permanecer documentado como diferença Notebook × Web até uma implementação compatível ser definida.

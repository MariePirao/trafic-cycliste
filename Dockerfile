FROM python:3.12
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY src .

EXPOSE 8501

# Setup an app user so the container doesn't run as the root user
RUN useradd app && \
    mkdir -p /home/app && \
    chown -R app:app /home/app && \
    mkdir -p /usr/local/app/data && \
    chown -R app:app /usr/local/app/data 

USER app

# Créer un dossier temporaire pour Matplotlib
# RUN mkdir -p /tmp/matplotlib && chmod -R 777 /tmp/matplotlib && chown app /tmp/matplotlib
# Définir la variable d’environnement
ENV MPLCONFIGDIR=/tmp/matplotlib

CMD ["streamlit", "run", "homePage.py"]

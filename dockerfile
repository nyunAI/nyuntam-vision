
FROM nyunadmin/nyun_kompress:main_kompress
RUN rm -r /workspace
COPY . /workspace
WORKDIR /workspace
RUN pip install strenum
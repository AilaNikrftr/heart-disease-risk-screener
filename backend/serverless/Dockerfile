# Lambda container image. scikit-learn + pandas are too large for a zip
# deployment (250 MB unzipped limit), so the functions ship as one image and
# each function picks its handler through ImageConfig.Command.
FROM public.ecr.aws/lambda/python:3.15

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY app ${LAMBDA_TASK_ROOT}/app
COPY ml ${LAMBDA_TASK_ROOT}/ml
COPY serverless ${LAMBDA_TASK_ROOT}/serverless

# Train on NHANES 2015-2016 while building, so the model ships inside the image.
RUN cd ${LAMBDA_TASK_ROOT} && python -m ml.fetch_data && python -m ml.train && rm -rf ml/data

CMD ["serverless.handlers.create_screening"]

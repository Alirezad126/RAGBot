#!/usr/bin/env bash
# Build the image, push it to ECR, and point the Lambda at it.
#
# The function is created on first run and updated after that. GROQ_API_KEY and
# ALLOWED_ORIGINS must be in the environment, or already set on the function.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
NAME="${FUNCTION_NAME:-ragbot}"
MEMORY="${MEMORY:-1024}"
TIMEOUT="${TIMEOUT:-45}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${NAME}"
ROLE_NAME="${NAME}-lambda"

aws ecr describe-repositories --repository-names "$NAME" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$NAME" --region "$REGION" >/dev/null

# Only the latest image is worth keeping: every push replaces the one before it.
aws ecr put-lifecycle-policy --repository-name "$NAME" --region "$REGION" \
  --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Keep the newest image only","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":1},"action":{"type":"expire"}}]}' \
  >/dev/null

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Lambda rejects the OCI manifest list that buildx produces by default, so
# provenance and SBOM attestations are off and the push is a single manifest.
docker build --platform linux/arm64 --provenance=false --sbom=false \
  -t "${REPO}:latest" .
docker push "${REPO}:latest"
DIGEST="$(aws ecr describe-images --repository-name "$NAME" --region "$REGION" \
  --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text)"

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  # No S3 and no VPC, so the basic execution role is the whole of it.
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting for the new role to propagate…"
  sleep 15
fi
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

# As JSON, not the key=value shorthand: ALLOWED_ORIGINS is comma-separated and
# the shorthand parser reads those commas as its own separators. Through a file
# so the key never appears in a process listing.
ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT
GROQ_API_KEY="${GROQ_API_KEY:?set GROQ_API_KEY}" \
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:?set ALLOWED_ORIGINS}" \
python3 -c 'import json, os; print(json.dumps({"Variables": {k: os.environ[k] for k in ("GROQ_API_KEY", "ALLOWED_ORIGINS")}}))' \
  > "$ENV_FILE"

if aws lambda get-function --function-name "$NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$NAME" --region "$REGION" \
    --image-uri "${REPO}@${DIGEST}" >/dev/null
  aws lambda wait function-updated --function-name "$NAME" --region "$REGION"
  aws lambda update-function-configuration --function-name "$NAME" --region "$REGION" \
    --memory-size "$MEMORY" --timeout "$TIMEOUT" \
    --environment "file://${ENV_FILE}" >/dev/null
else
  aws lambda create-function --function-name "$NAME" --region "$REGION" \
    --package-type Image --code "ImageUri=${REPO}@${DIGEST}" \
    --role "$ROLE_ARN" --architectures arm64 \
    --memory-size "$MEMORY" --timeout "$TIMEOUT" \
    --environment "file://${ENV_FILE}" >/dev/null
  aws lambda wait function-active-v2 --function-name "$NAME" --region "$REGION"
  # CORS is handled by the app, so the function URL just has to let it through.
  aws lambda create-function-url-config --function-name "$NAME" --region "$REGION" \
    --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --region "$REGION" \
    --statement-id public-function-url --action lambda:InvokeFunctionUrl \
    --principal '*' --function-url-auth-type NONE >/dev/null
  # Both actions, or the URL answers 403. The URL is already unauthenticated,
  # so this widens nothing that was not already open.
  aws lambda add-permission --function-name "$NAME" --region "$REGION" \
    --statement-id public-invoke --action lambda:InvokeFunction \
    --principal '*' >/dev/null
fi

aws lambda wait function-updated --function-name "$NAME" --region "$REGION"
# Logs are the only thing here that grows without a bound, so cap their age.
aws logs put-retention-policy --region "$REGION" \
  --log-group-name "/aws/lambda/${NAME}" --retention-in-days 14 >/dev/null 2>&1 || true

echo
echo "Function URL:"
aws lambda get-function-url-config --function-name "$NAME" --region "$REGION" \
  --query FunctionUrl --output text

FROM node:22.19.0-alpine AS build

WORKDIR /frontend
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM nginx:1.29.2-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /frontend/dist /usr/share/nginx/html
EXPOSE 80

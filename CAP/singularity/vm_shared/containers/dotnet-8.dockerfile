FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update
RUN apt-get install -y --no-install-recommends wget apt-transport-https software-properties-common

RUN wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb 
RUN dpkg -i packages-microsoft-prod.deb
RUN apt-get update
RUN apt-get install -y --no-install-recommends dotnet-sdk-8.0

RUN echo "echo \"<Project Sdk=\\\"Microsoft.NET.Sdk\\\"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\" > \${1%.*}.csproj\ndotnet build \${1%.*}.csproj \${@:3}\nmv \$(dirname \$1)/bin/Debug/net8.0/\$(basename \${1%.*}).dll \$2" > /app/dotnet-build
RUN chmod 777 /app/dotnet-build
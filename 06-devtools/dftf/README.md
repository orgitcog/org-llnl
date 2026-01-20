Drink From The Firehose (DFTF) is a python program that subscribes to
Redfish events on Cray/HPE hardware, and republishes them to topics
in Kafka. Additionally, it handles Slingshot Telemetry messages with
a MessageID prefix of CrayFabricHealth.

Slingshot telemetry must be configured separately using slingshot
CLI tools.

To create Redfish subscription, use the dftfd_subscriber daemon.
It will create two subscriptions on each configured Redfish server:

- CrayTelemetry Alerts
- Normal (non-CrayTelemetry) Alerts

CrayTelemetry alerts are sent at regular intervals (typically 5 seconds)
decided upon by the Redfish server. The OEM field of the event contains
a large amount of metric data formated in a nonstandard (OEM decided)
Cray format. Typically the event message consolidates metric samples
taken at one second intervals.

Topic names for Kafka have a common configurable prefix. The rest of
the topic name for the CrayTelemetry data is "craytelemetry", and the
rest of the topic name for normal Alerts is "crayevents".

DFTF is made up of two main daemons: dftfd and dftfd_subscriber.

dftfd sets up an http server, processes incoming redfish (and slinghost FA/FMN)
alert messages, and publishes metrics to kafka topics.

dftfd_subscriber connects to Redfish servers at regular intervals, creating
or refreshing redfish subscriptions that point to the various dftfd that
you have set up on your system.

See the dftfd.yaml.example and dftfd_subscriber.yaml.example for example
configurations.

License
----------------

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-849577

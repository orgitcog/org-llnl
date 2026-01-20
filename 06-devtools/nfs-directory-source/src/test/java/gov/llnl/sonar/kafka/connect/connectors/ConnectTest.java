// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import lombok.extern.log4j.Log4j2;
import org.apache.avro.generic.GenericData;
import org.apache.kafka.clients.consumer.Consumer;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Log4j2
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public abstract class ConnectTest {

    ConfluentDriver confluent;

    @BeforeAll
    public void setup() {
        confluent = new ConfluentDriver("/Users/gimenez1/Home/local/src/confluent-5.0.1/bin");
    }

    @AfterAll
    public void teardown() {
        confluent.close();
    }

    void validateTopicContents(String topic, Set<GenericData.Record> trueData) {

        log.info("Consuming topic " + topic);

        List<GenericData.Record> consumedRecords = new ArrayList<>();
        Consumer consumer = confluent.createConsumer(topic);

        while (consumedRecords.size() < trueData.size()) {

            Iterable<ConsumerRecord> consumerStream = consumer.poll(Duration.ofSeconds(10));
            for (ConsumerRecord consumerRecord : consumerStream) {

                // Parse to avro record
                GenericData.Record record = (GenericData.Record) consumerRecord.value();
                log.info("<<< Consumed record: " + record.toString());

                consumedRecords.add(record);

            }
        }

        consumer.close();

        assertEquals(trueData, new HashSet<>(consumedRecords));
    }


}

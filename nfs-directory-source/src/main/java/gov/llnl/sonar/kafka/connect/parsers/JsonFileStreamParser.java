// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.parsers;

import gov.llnl.sonar.kafka.connect.converters.ConvertException;
import gov.llnl.sonar.kafka.connect.offsetmanager.FileOffset;
import io.confluent.connect.avro.AvroData;
import lombok.extern.slf4j.Slf4j;
import org.apache.avro.AvroTypeException;
import org.apache.avro.generic.GenericData;
import org.apache.avro.io.Decoder;
import org.apache.avro.io.DecoderFactory;
import org.apache.avro.specific.SpecificDatumReader;
import org.apache.kafka.connect.source.SourceRecord;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Path;

@Slf4j
public class JsonFileStreamParser extends FileStreamParser {

    private SpecificDatumReader<GenericData.Record> datumReader;

    public JsonFileStreamParser(Path filePath,
                               JSONObject formatOptions,
                               AvroData avroData,
                               org.apache.avro.Schema avroSchema,
                               org.apache.kafka.connect.data.Schema connectSchema,
                               String eofSentinel,
                               int bufferSize,
                               FileOffset offset,
                               String partitionField,
                               String offsetField) throws IOException {
        super(filePath,
              formatOptions,
              avroData,
              avroSchema,
              connectSchema,
              eofSentinel,
              bufferSize,
              offset,
              partitionField,
              offsetField);

        datumReader = new SpecificDatumReader<>(avroSchema);
    }

    @Override
    public synchronized SourceRecord readNextRecord(String topic)
            throws EOFException, ParseException, ConvertException, IOException {

        GenericData.Record datum = new GenericData.Record(avroSchema);
        try {
            Decoder decoder = DecoderFactory.get().jsonDecoder(avroSchema, nextLine());
            datum = datumReader.read(datum, decoder);
        } catch (AvroTypeException e) {
            throw new ParseException(this, e.getMessage());
        }

        try {
            return new SourceRecord(
                    sourcePartition,
                    getSourceOffset(),
                    topic,
                    connectSchema,
                    avroData.toConnectData(avroSchema, datum).value());
        } catch (Exception e) {
            throw new ConvertException(datum, avroSchema.toString(true), e.getMessage());
        }
    }
}

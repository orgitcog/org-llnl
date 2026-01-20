// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.converters;

import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.connect.data.Schema;
import org.apache.kafka.connect.data.Struct;
import org.apache.kafka.connect.errors.DataException;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Class for performing conversions from a set of string tokens parsed from a CSV file to a Kafka Connect record.
 */
@Slf4j
public class CsvRecordConverter {

    private final Schema connectSchema;
    private final String[] columns;

    /**
     * Constructor.
     *
     * @param schema The Kafka Connect schema of the columns to convert into
     * @param columns The ordered names of columns for the incoming string tokens
     */
    public CsvRecordConverter(Schema schema, String[] columns) {
        this.connectSchema = schema;
        this.columns = columns;
    }

    /**
     * Converter for a single string token.
     *
     * @param s The string token to convert
     * @param type The Kafka Connect schema type to convert into
     * @return The typed Kafka Connect token
     */
    public Object stringToConnectObject(String s, Schema.Type type) {

        switch (type) {
            case STRING:
                return s;
            case INT8:
                return Byte.valueOf(s);
            case INT16:
                return Short.valueOf(s);
            case INT32:
                return Integer.valueOf(s);
            case INT64:
                return Long.valueOf(s);
            case FLOAT32:
                return Float.valueOf(s);
            case FLOAT64:
                return Double.valueOf(s);
            case BOOLEAN:
                return Boolean.valueOf(s);
            case BYTES:
                return ByteBuffer.wrap(s.getBytes());
            case STRUCT:
            case MAP:
            case ARRAY:
                throw new DataException("Non-primitive types not supported for CSV file sources!");
        }

        return null;
    }

    /**
     * Converter for the string token array (a single CSV row).
     *
     * @param csvTokens The string token array to convert
     * @return The Kafka Connect record with the schema provided in the constructor
     */
    public Struct convert(String[] csvTokens) throws ConvertException {
        Struct record = new Struct(connectSchema);

        int i = 0;
        for (String column : columns) {

            String value = csvTokens[i++];

            try {
                Object parsedValue = stringToConnectObject(value, connectSchema.field(column).schema().type());
                record = record.put(column, parsedValue);
            } catch (NumberFormatException e) {
                throw new ConvertException(value, connectSchema.field(column).schema().type(), e.getMessage());
            } catch (NullPointerException e) {
                throw new ConvertException(value, column, e.getMessage());
            }
        }

        return record;
    }
}

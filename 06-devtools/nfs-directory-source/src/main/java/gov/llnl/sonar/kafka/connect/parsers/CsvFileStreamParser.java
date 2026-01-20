// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.parsers;

import gov.llnl.sonar.kafka.connect.converters.ConvertException;
import gov.llnl.sonar.kafka.connect.converters.CsvRecordConverter;
import gov.llnl.sonar.kafka.connect.offsetmanager.FileOffset;
import io.confluent.connect.avro.AvroData;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.connect.source.SourceRecord;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Path;
import java.text.MessageFormat;
import java.util.*;

@Slf4j
public class CsvFileStreamParser extends FileStreamParser {

    // CSV semantics
    private int numColumns;
    private String[] columns = null;
    private int delimChar = (int)',';
    private int commentChar = (int)'#';
    private boolean hasHeader = true;

    // Vars for parsing
    private StringBuilder sb = new StringBuilder();
    private CsvRecordConverter csvRecordConverter;

    public CsvFileStreamParser(Path filePath,
                               JSONObject formatOptions,
                               AvroData avroData,
                               org.apache.avro.Schema avroSchema,
                               org.apache.kafka.connect.data.Schema connectSchema,
                               String eofSentinel,
                               int bufferSize,
                               FileOffset offset,
                               String partitionField,
                               String offsetField) throws ParseException, IOException {
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

        parseCsvFormatOptions(formatOptions);
        parseHeader();

        csvRecordConverter = new CsvRecordConverter(connectSchema, columns);
    }

    private synchronized void parseHeader() throws ParseException, IOException {
        if (columns == null && hasHeader) {
            columns = readCsvLineIntoTokens();
            numColumns = columns.length;
        } else if (columns == null) {
            throw new IllegalArgumentException("No header and no columns specified for CSV parser!");
        }
    }

    private void parseCsvFormatOptions(JSONObject formatOptions) {
        for (String option : formatOptions.keySet()) {
            switch (option) {
                case ("withHeader"):
                    hasHeader = formatOptions.getBoolean(option);
                    break;
                case ("columns"):
                    JSONArray arr = formatOptions.getJSONArray(option);
                    columns = new String[arr.length()];
                    for(int i = 0; i < arr.length(); i++){
                        columns[i] = arr.getString(i);
                    }
                    numColumns = columns.length;
                    break;
                case ("delimiter"):
                    delimChar = formatOptions.getString(option).charAt(0);
                    break;
                case ("commentChar"):
                    commentChar = formatOptions.getString(option).charAt(0);
                    break;
            }
        }
    }

    private synchronized String[] readCsvLineIntoTokens() throws ParseException, IOException {

        if (bufferedReader == null) {
            throw new EOFException("Reader closed!");
        }

        // Container for tokens to return
        ArrayList<String> lineTokens;
        if (numColumns > 0) {
            lineTokens = new ArrayList<>(numColumns);
        } else {
            lineTokens = new ArrayList<>();
        }

        // Reset StringBuilder
        sb.setLength(0);
        boolean firstChar = true;
        while (true) {
            int c = bufferedReader.read();
            offset.incrementByteOffset(1L);

            if (c == -1) {
                throw new EOFException(MessageFormat.format(
                        "EOF encountered at file {0}, offset {1}",
                        fileName, offset.toString()));
            } else if (firstChar && c == commentChar) {
                // Commented line, skip
                skipLine();
            } else if (c == newLineChar) {
                // Build token, reset StringBuilder, stop reading
                lineTokens.add(sb.toString());
                sb.setLength(0);
                break;
            } else if (c == delimChar) {
                // Build token, reset StringBuilder
                lineTokens.add(sb.toString());
                sb.setLength(0);
            } else if (c == 0) {
                // Null character, don't add
            } else {
                // Add char to current StringBuilder
                sb.append((char) c);
            }

            firstChar = false;
        }

        // Construct array
        String[] lineTokensArray = lineTokens.toArray(new String[lineTokens.size()]);

        // Check length of tokens against columns
        if (numColumns > 0 && lineTokens.size() != numColumns) {
            throw new ParseException(this, "Invalid number of columns");
        }

        // Return tokens as String[]
        return lineTokensArray;
    }

    @Override
    public synchronized SourceRecord readNextRecord(String topic)
            throws EOFException, ParseException, ConvertException, IOException {
        return new SourceRecord(
                sourcePartition,
                getSourceOffset(),
                topic,
                connectSchema,
                csvRecordConverter.convert(readCsvLineIntoTokens()));
    }
}


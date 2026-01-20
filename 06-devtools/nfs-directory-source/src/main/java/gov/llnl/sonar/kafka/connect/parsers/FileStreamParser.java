// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.parsers;

import gov.llnl.sonar.kafka.connect.converters.ConvertException;
import gov.llnl.sonar.kafka.connect.offsetmanager.FileOffset;
import io.confluent.connect.avro.AvroData;
import lombok.extern.slf4j.Slf4j;
import org.apache.avro.Schema;
import org.apache.kafka.connect.source.SourceRecord;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.MessageFormat;
import java.util.Collections;
import java.util.Map;

@Slf4j
public abstract class FileStreamParser {

    // File semantics
    String fileName;
    Path filePath;
    JSONObject formatOptions;
    String eofSentinel;

    // Schema semantics
    AvroData avroData;
    org.apache.avro.Schema avroSchema;
    org.apache.kafka.connect.data.Schema connectSchema;

    // File reading vars
    int bufferSize;
    FileOffset offset;
    RandomAccessFile randomAccessFile;
    FileInputStream fileInputStream;
    BufferedReader bufferedReader;
    final int newLineChar = (int) '\n';

    // Kafka vars
    String partitionField;
    String offsetField;
    Map<String, String> sourcePartition;

    // Abstract functions
    public abstract SourceRecord readNextRecord(String topic)
            throws EOFException, ParseException, ConvertException, IOException;

    public FileStreamParser(Path filePath,
                            JSONObject formatOptions,
                            AvroData avroData,
                            Schema avroSchema,
                            org.apache.kafka.connect.data.Schema connectSchema,
                            String eofSentinel,
                            int bufferSize,
                            FileOffset offset,
                            String partitionField,
                            String offsetField) throws IOException {
        this.filePath = filePath;
        this.fileName = filePath.toString();
        this.formatOptions = formatOptions;
        this.avroData = avroData;
        this.avroSchema = avroSchema;
        this.connectSchema = connectSchema;
        this.eofSentinel = eofSentinel;
        this.bufferSize = bufferSize;
        this.offset = offset;
        this.partitionField = partitionField;
        this.offsetField = offsetField;

        this.sourcePartition = Collections.singletonMap(partitionField, fileName);

        seekToOffset(offset);
    }

    public String getFileName() {
        return fileName;
    }

    public Path getFilePath() {
        return filePath;
    }

    public Map<String, Long> getSourceOffset() {
        return Collections.singletonMap(offsetField, offset.getByteOffset());
    }

    public synchronized void deleteFile() throws IOException {
        Files.delete(filePath);
    }

    /**
     * Moves the current file found within sourceDir to destDir with subdirectories.
     * If file is /a/b/c.txt, sourceDir is /a and destDir is /x, will create
     * subdirectory /x/b and move file to /x/b/c.txt
     *
     * @param sourceDir
     * @param destDir
     * @throws IOException
     */
    public synchronized void moveFileIntoDirectory(Path sourceDir, Path destDir) throws IOException {
        Path relativePathToFile = sourceDir.relativize(filePath).normalize();
        Path completedFilePath = destDir.resolve(relativePathToFile).normalize();
        Path completedFileParentPath = completedFilePath.getParent();
        Files.createDirectories(completedFileParentPath);
        Files.move(filePath, completedFilePath);
    }

    public FileOffset getOffset() {
        return offset;
    }

    public void unlock() {
        offset.setLocked(false);
    }

    public void complete(boolean delete, Path dirPath, Path completedDirPath) throws IOException {
        offset.setLocked(true);
        offset.setCompleted(true);

        if (delete) {
            Files.delete(filePath);
        } else {
            moveFileIntoDirectory(dirPath, completedDirPath);
        }
    }

    public synchronized void seekToOffset(FileOffset offset) throws IOException {
        close();

        // Random access at offset
        randomAccessFile = new RandomAccessFile(fileName, "r");
        randomAccessFile.seek(offset.getByteOffset());

        this.offset = offset;

        // Create input stream and reader at seek'ed file
        fileInputStream = new FileInputStream(randomAccessFile.getFD());
        bufferedReader = new BufferedReader(new InputStreamReader(fileInputStream), bufferSize);
    }

    synchronized void skipLine() throws IOException {
        int c;
        do {
            c = bufferedReader.read();
            offset.incrementByteOffset(1L);
            if (c == -1) {
                throw new EOFException("End of file reached!");
            }
        } while (c != newLineChar);
        offset.incrementLineOffset(1L);
    }

    synchronized String nextLine() throws IOException {
        try {

            if (bufferedReader == null) {
                throw new EOFException("Reader closed!");
            }

            String lineString = bufferedReader.readLine();

            if (lineString == null || (eofSentinel != null && lineString.equals(eofSentinel))) {
                throw new EOFException(MessageFormat.format(
                        "EOF sentinel encountered at file {0}, offset {1}",
                        fileName, offset.toString()));
            }

            offset.incrementByteOffset(lineString.getBytes().length + 1);
            offset.incrementLineOffset(1L);

            return lineString;

        } catch (EOFException e) {
            close();
            throw new EOFException(MessageFormat.format(
                    "EOF encountered at file {0}, offset {1}",
                    fileName, offset.toString()));
        }
    }

    public synchronized void close() throws IOException {
        if (bufferedReader != null) {
            bufferedReader.close();
            bufferedReader = null;
        }
        if (fileInputStream != null) {
            fileInputStream.close();
            fileInputStream = null;
        }
        if (randomAccessFile != null) {
            randomAccessFile.close();
            randomAccessFile = null;
        }
    }
}

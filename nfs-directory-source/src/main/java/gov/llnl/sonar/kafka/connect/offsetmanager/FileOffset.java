// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.offsetmanager;

import java.io.Serializable;

/**
 * Represents the current read state of a file.
 * Instances of FileOffset are meant to be serialized into Zookeeper nodes (via FileOffsetManager) for concurrent access.
 */
public class FileOffset implements Serializable {

    /**
     * Current read offset, in bytes.
     */
    long byteOffset;

    /**
     * Current read byteOffset, in lines.
     */
    long lineOffset;

    /**
     * Whether the file is currently locked for reading.
     */
    boolean locked;

    /**
     * Whether the file has been fully read.
     */
    boolean completed;

    private static final long serialVersionUID = 1L;

    public FileOffset(long byteOffset, long lineOffset, boolean locked, boolean completed) {
        this.byteOffset = byteOffset;
        this.lineOffset = lineOffset;
        this.locked = locked;
        this.completed = completed;
    }

    @Override
    public String toString() {
        return String.format("FileOffset(byteOffset=%d, lineOffset=%d, locked=%b, completed=%b)",
                byteOffset, lineOffset, locked, completed);
    }

    public void incrementByteOffset(long bytes) {
        this.byteOffset += bytes;
    }

    public void incrementLineOffset(long lines) {
        this.lineOffset += lines;
    }

    public void setLocked(boolean locked) {
        this.locked = locked;
    }

    public void setCompleted(boolean completed) {
        this.completed = completed;
    }

    public boolean getCompleted() {
        return completed;
    }

    public long getByteOffset() {
        return byteOffset;
    }

    public long getLineOffset() {
        return byteOffset;
    }
}


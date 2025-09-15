package com.example.facerecognizationdemo

interface SimilarityClassifier {

    /** An immutable result returned by a Classifier describing what was recognized. */
    class Recognition(
        val id: String?,       // unique identifier
        val title: String?,    // display name
        val distance: Float?   // similarity distance
    ) {
        var extra: Any? = null

        override fun toString(): String {
            val result = StringBuilder()
            if (id != null) {
                result.append("[$id] ")
            }
            if (title != null) {
                result.append("$title ")
            }
            if (distance != null) {
                result.append(String.format("(%.1f%%) ", distance * 100.0f))
            }
            return result.toString().trim()
        }
    }
}

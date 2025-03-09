import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  ActivityIndicator,
  StyleSheet,
  Alert,
  ScrollView,
  Platform,
} from "react-native";
import { useRouter } from "expo-router";
import Icon from "react-native-vector-icons/Ionicons";
import Speedometer from "react-native-speedometer-chart";

export default function Score() {
  const router = useRouter();
  const [sourceAddress, setSourceAddress] = useState("Fetching location...");
  const [destination, setDestination] = useState("");
  const [loading, setLoading] = useState(false);
  const [animatedScore, setAnimatedScore] = useState(0);
  const [finalScore, setFinalScore] = useState(null);
  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {
    const fetchLocation = async () => {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;
          await fetchAddressFromCoords(latitude, longitude);
        },
        (error) => {
          console.error("Failed to fetch location:", error);
          setSourceAddress("Failed to fetch location");
        }
      );
    };
    fetchLocation();
  }, []);

  const fetchAddressFromCoords = async (latitude, longitude) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
      );
      const data = await response.json();
      setSourceAddress(data.display_name || "Location not found");
    } catch {
      setSourceAddress("Failed to fetch address");
    }
  };

  const handleCheckScore = async () => {
    if (!destination) {
      Alert.alert("Error", "Destination is required.");
      return;
    }
    setLoading(true);
    setSuggestions([]);
    setFinalScore(null);
    setAnimatedScore(1);

    try {
      const url =
        Platform.OS === "android"
          ? "http://127.0.0.1:8081/predict-risk"
          : "http://127.0.0.1:8081/predict-risk";

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sloc: sourceAddress, des: destination }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const score = data.risk_score || 0;
      setSuggestions(data.suggestions || []);
      animateSpeedometer(score);
    } catch (error) {
      console.error("Fetch error:", error);
      Alert.alert("Error", "Failed to fetch risk score.");
    }

    setLoading(false);
  };

  const animateSpeedometer = (finalScoreValue) => {
    setFinalScore(finalScoreValue);
    let currentScore = animatedScore;
    const step = finalScoreValue > currentScore ? 1 : -1;
    const interval = setInterval(() => {
      currentScore += step;
      setAnimatedScore(Math.min(Math.max(currentScore, 0), finalScoreValue));
      if (
        (step > 0 && currentScore >= finalScoreValue) ||
        (step < 0 && currentScore <= finalScoreValue)
      ) {
        clearInterval(interval);
      }
    }, 50);
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContent}>
      <View style={styles.container}>
        <Pressable
          style={styles.backButton}
          onPress={() => router.push("/home")}
        >
          <Icon name="arrow-back" size={25} color="#1E3A8A" />
          <Text style={styles.backText}>Back</Text>
        </Pressable>

        <Text style={styles.title}>Risk Score</Text>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>Source (Your Location):</Text>
          <Text style={styles.locationText}>{sourceAddress}</Text>
        </View>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>Destination:</Text>
          <TextInput
            style={styles.input}
            value={destination}
            onChangeText={setDestination}
            placeholder="Enter destination"
            placeholderTextColor="#999"
          />
        </View>

        <Pressable
          style={styles.mapButton}
          onPress={() => {
            const destinationQuery = encodeURIComponent(destination || "");
            const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${destinationQuery}`;
            router.push(mapsUrl);
          }}
        >
          <Text style={styles.mapButtonText}>Open in Google Maps</Text>
        </Pressable>

       

        <Pressable onPress={handleCheckScore} style={styles.button}>
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Check Risk Score</Text>
          )}
        </Pressable>

        {finalScore !== null && (
          <View style={styles.gaugeContainer}>
            <Speedometer
              value={animatedScore}
              totalValue={10}
              size={200}
              outerColor="#e0e0e0"
              internalColor={animatedScore > 7 ? "#d32f2f" : "#388e3c"}
            />
            <Text style={styles.gaugeText}>
              Risk Score: {animatedScore.toFixed(1)}/10
            </Text>
          </View>
        )}

        {suggestions.length > 0 && (
          <>
            <Text style={styles.suggestionsTitle}>
              Suggestions for Safer Cycling:
            </Text>
            {suggestions.map((suggestion, index) => (
              <Text key={index} style={styles.suggestionText}>
                • {suggestion.replace(/\*\*/g, "")}
              </Text>
            ))}
          </>
        )}
         <Pressable onPress={() => router.push("/sos")} style={styles.sosButton}>
          <Text style={styles.sosButtonText}>SOS</Text>
        </Pressable>
      </View>
     
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollContent: {
    paddingBottom: 20,
  },
  locationText:{
fontSize:20
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: 60,
  },
  backButton: {
    position: "absolute",
    top: 40,
    left: 20,
    flexDirection: "row",
    alignItems: "center",
    zIndex: 10,
  },
  backText: {
    marginLeft: 5,
    fontSize: 20,
    color: "#1E3A8A",
  },
  title: {
    fontSize: 26,
    fontWeight: "bold",
    color: "#264653",
    textAlign: "center",
    marginBottom: 24,
    marginTop: 16,
  },
  inputContainer: {
    marginBottom: 16,
  },
  label: {
    fontSize: 20,
    color: "#1e40af",
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: "#94d2bd",
    borderRadius: 8,
    padding: 12,
    fontSize: 18,
    color: "#005f73",
    backgroundColor: "#f1faff",
  },
  button: {
    backgroundColor: "#0d6efd",
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 16,
  },
  buttonText: {
    color: "#ffffff",
    fontWeight: "bold",
    fontSize: 18,
  },
  mapButton: {
    backgroundColor: "#4caf50",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 16,
  },
  mapButtonText: {
    color: "#ffffff",
    fontWeight: "bold",
    fontSize: 18,
  },
  gaugeContainer: {
    alignItems: "center",
    marginVertical: 20,
    backgroundColor: "#ffffff",
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: "#dee2e6",
  },
  gaugeText: {
    fontSize: 18,
    color: "#212529",
    marginTop: 10,
    fontWeight: "bold",
  },
  suggestionsTitle: {
    color: "#0d6efd",
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 8,
  },
  suggestionText: {
    color: "#0d47a1",
    fontSize: 22,
    marginBottom: 12,
  },
  sosButton: {
    backgroundColor: "#dc3545",
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 16,
  },
  sosButtonText: {
    color: "#ffffff",
    fontWeight: "bold",
    fontSize: 18,
  },
});

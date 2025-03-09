import { useState, useEffect } from "react";
import { View, Text, Pressable, ActivityIndicator, StyleSheet, Alert, SafeAreaView } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Icon from "react-native-vector-icons/Ionicons";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";

export default function SOSButton() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [sosCalled, setSosCalled] = useState(false);
  const [userId, setUserId] = useState(null);

  useEffect(() => {
    const fetchUserId = async () => {
      try {
        const storedId = await AsyncStorage.getItem("userId");
        if (storedId) setUserId(storedId);
      } catch (error) {
        console.error("Error fetching userId:", error);
      }
    };
    fetchUserId();
  }, []);

  const handleSOS = async () => {
    setLoading(true);
    const payload = {
      userId: userId,
      from: "+17653454972",
    };

    try {
      const response = await fetch("http://10.10.119.148:8081/user/sos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (response.ok) {
        setSosCalled(true);
      } else {
        Alert.alert("Error", data.error || "Failed to send SOS.");
      }
    } catch (error) {
      Alert.alert("Error", "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    
    <SafeAreaView style={styles.safeArea}>
        <Pressable style={styles.backButton} onPress={() => router.replace("/home")}>
          <Icon name="arrow-back" size={24} color="1E3A8A" />
          <Text style={styles.backText}>Back</Text>
        </Pressable>

        <View style={styles.content}>
          {loading ? (
            <ActivityIndicator size="large" color="#22d3ee" />
          ) : sosCalled ? (
            <View style={styles.successContainer}>
              <Icon name="checkmark-circle-outline" size={80} color="#22c55e" />
              <Text style={styles.successText}>SOS Called Successfully</Text>
            </View>
          ) : (
            <Pressable onPress={handleSOS} style={styles.button}>
              <Text style={styles.buttonText}>Send SOS</Text>
            </Pressable>
          )}
        </View>
      
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#f8f9fa",
  },
  container: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
  },
  backButton: {
    position: "absolute",
    top: 50,
    left: 20,
    flexDirection: "row",
    alignItems: "center",
    zIndex: 10,
  },
  backText: {
    marginLeft: 5,
    fontSize: 16,
    color: "#1E3A8A",
  },
  content: {
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
  },
  button: {
    backgroundColor: "#dc2626",
    paddingVertical: 16,
    paddingHorizontal: 10,
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    width: "100%",
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 18,
  },
  successContainer: {
    alignItems: "center",
    marginTop: -40,
  },
  successText: {
    color: "#22c55e",
    fontSize: 24,
    fontWeight: "bold",
    marginTop: 16,
  },
});

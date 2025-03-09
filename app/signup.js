import { useState } from "react";
import { View, TextInput, Text, Pressable, ActivityIndicator, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { LinearGradient } from "expo-linear-gradient";
import Icon from "react-native-vector-icons/Ionicons";

export default function Signup() {
  const router = useRouter();
  const [form, setForm] = useState({
    email: "",
    username: "",
    password: "",
    mobile_no: "",
    alternate_no: "",
  });

  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleSignup = async () => {
    setLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch("http://10.10.119.148:8081/user/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        setErrorMessage(data?.error || "Signup failed");
        return;
      }

      if (data.userId && data.userId.length === 24) {
        await storeUserId(data.userId);
        router.replace("/home");
      } else {
        setErrorMessage("Invalid userId received.");
      }
    } catch (error) {
      setLoading(false);
      setErrorMessage("Something went wrong. Please try again.");
    }
  };

  const storeUserId = async (userId) => {
    try {
      await AsyncStorage.setItem("userId", userId);
    } catch (error) {
      console.error("Error storing user ID:", error);
    }
  };

  return (
    
 <LinearGradient
    colors={["#1E3A8A", "#000000", ]}
      style={styles.container}
    >      <Pressable style={styles.backButton} onPress={() => router.push("/")}>
              <Icon name="arrow-back" size={24} color="white" />
              <Text style={styles.backText}>Back</Text>
            </Pressable>
      
      <Text style={styles.title}>Signup</Text>

      <TextInput
        style={styles.input}
        placeholder="Email"
        placeholderTextColor="#888"
        autoCapitalize="none"
        value={form.email}
        onChangeText={(text) => setForm({ ...form, email: text })}
      />

      <TextInput
        style={styles.input}
        placeholder="Username"
        placeholderTextColor="#888"
        value={form.username}
        onChangeText={(text) => setForm({ ...form, username: text })}
      />

      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor="#888"
        secureTextEntry
        value={form.password}
        onChangeText={(text) => setForm({ ...form, password: text })}
      />

      <TextInput
        style={styles.input}
        placeholder="Mobile Number"
        placeholderTextColor="#888"
        keyboardType="phone-pad"
        value={form.mobile_no}
        onChangeText={(text) => setForm({ ...form, mobile_no: text })}
      />

      <TextInput
        style={styles.input}
        placeholder="SOS Mobile Number"
        placeholderTextColor="#888"
        keyboardType="phone-pad"
        value={form.alternate_no}
        onChangeText={(text) => setForm({ ...form, alternate_no: text })}
      />

      {errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}

      <Pressable onPress={handleSignup} disabled={loading} style={[styles.button, loading && styles.buttonDisabled]}>
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Sign Up</Text>
        )}
      </Pressable>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#ffffff",
    marginBottom: 32,
    textAlign: "center",
  },
  input: {
    borderWidth: 1,
    borderColor: "#fffff",
    color: "#ffffff",
    padding: 16,
    width: "100%",
    borderRadius: 12,
    marginBottom: 16,
    fontSize: 16,
    backgroundColor: "rgba(255, 255, 255, 0.1)",
  },
  errorText: {
    color: "#f87171",
    fontSize: 14,
    marginBottom: 16,
    textAlign: "center",
  },
  button: {
    backgroundColor: "#3b82f6",
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
    width: "100%",
    alignItems: "center",
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 18,
    fontWeight: "600",
  },
  backButton: {
    position: "absolute",
    top: 50,
    left: 20,
    flexDirection: "row",
    alignItems: "center",
  },
  backText: {
    marginLeft: 5,
    fontSize: 16,
    color: "white",
  },
});

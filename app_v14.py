
        # Gemini çağrısından ÖNCE öğrenci mesajını kalıcı kontrol
        # noktasına yaz.
        _save_chat_backup(student_id)


        # ----------------------------------------------------
        # GEMINI YANITI
        # ----------------------------------------------------

        # V29: Basamak geçişi yalnızca öğrenci açıkça tamamladığını söylediğinde yapılır.
        if _explicit_stage_completion(student_message):
            stages = list(BASAMAK_TALIMATLARI.keys())
            stage_index = stages.index(current_stage)
            if current_stage not in st.session_state.completed_stages:
                st.session_state.completed_stages.append(current_stage)
            log_kaydet({
                "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": student_id,
                "problem_session_id": st.session_state.get("problem_session_id"),
                "basamak": current_stage,
                "alt_beceri": st.session_state.last_alt_beceri.get(current_stage, ""),
                "basamak_tamamlandi": True,
                "tip": "Basamak Geçişi",
                "icerik": f"Öğrenci açıkça tamamladı: {current_stage}"
            })
            _save_chat_backup(student_id)
            if stage_index < len(stages) - 1:
                st.session_state.current_step = stages[stage_index + 1]
            st.rerun()

        with st.spinner(
            "Rehber Bot düşünüyor..."
        ):

            try:

                result = ogrenciye_cevap_ver(
                    current_stage,
                    st.session_state.problem_analysis,
                    st.session_state.chat_storage[current_stage][:-1],
                    student_message
                )

                answer = result.get("mesaj", result.get("soru", ""))
                alt_beceri = result["alt_beceri"]
                stage_completed = result["basamak_tamamlandi"] and _explicit_stage_completion(student_message)

                st.session_state.last_alt_beceri[current_stage] = alt_beceri

                st.session_state.chat_storage[current_stage].append({
                    "role": "assistant",
                    "content": answer,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed
                })

                log_kaydet({
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": student_id,
                    "problem_session_id": st.session_state.get("problem_session_id"),
                    "basamak": current_stage,
                    "alt_beceri": alt_beceri,
                    "basamak_tamamlandi": stage_completed,
                    "müdahale_tipi": result.get("müdahale_tipi", ""),
                    "tip": "Bot",
                    "icerik": answer
                })

                student_turns = sum(
                    1 for m in st.session_state.chat_storage[current_stage]
                    if m["role"] == "user"
                )

                stages = list(BASAMAK_TALIMATLARI.keys())
                stage_index = stages.index(current_stage)

                _save_chat_backup(student_id)

                if stage_completed and student_turns >= 2:
                    if current_stage not in st.session_state.completed_stages:
                        st.session_state.completed_stages.append(current_stage)

                    if stage_index < len(stages) - 1:
                        next_stage = stages[stage_index + 1]
                        st.session_state.current_step = next_stage

                        log_kaydet({
                            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "id": student_id,
                            "basamak": current_stage,
                            "alt_beceri": alt_beceri,
                            "basamak_tamamlandi": True,
                            "tip": "Basamak Geçişi",
                            "icerik": f"{current_stage} tamamlandı → {next_stage}"
                        })

                st.rerun()


            except Exception as e:

                st.error(
                    "Gemini yanıt oluşturamadı."
                )

                st.code(
                    str(e)
                )
